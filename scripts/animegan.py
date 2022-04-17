import os
import sys
sys.path.insert(0, os.getcwd())
import pytorch_lightning as pl
from networks.gan import AnimeGeneratorLite, AnimeDiscriminator, UnetGenerator, SpectNormDiscriminator, AnimeGenerator
from networks.pretrainnet import VGGPreTrained, VGGCaffePreTrained
from datamodules.animegands import AnimeGANDataModule
from datamodules.dsfunction import denormalize
from losses.gan_loss import LSGanLoss
from losses.lsfunction import variation_loss, rgb2yuv
import torch
import torch.nn as nn
from scripts.common import run_common, log_images
from typing import *

class AnimeGAN(pl.LightningModule):
  GeneratorDict = {
      'AnimeGenerator': AnimeGenerator,
      'AnimeGeneratorLite': AnimeGeneratorLite,
      'UnetGenerator': UnetGenerator,
  }
  DiscriminatorDict = {
      'AnimeDiscriminator': AnimeDiscriminator,
      'SpectNormDiscriminator': SpectNormDiscriminator,
  }

  PreTrainedDict = {
      'VGGPreTrained': VGGPreTrained,
      'VGGCaffePreTrained': VGGCaffePreTrained
  }

  def __init__(
      self,
      lr_g: float = 2e-4,
      lr_d: float = 2e-4,
      g_adv_weight: float = 300.,
      d_adv_weight: float = 300.,
      con_weight: float = 1.5,
      sty_weight: float = 2.8,
      color_weight: float = 10.,
      pre_trained_ckpt: str = None,
      generator_name: str = 'AnimeGeneratorLite',
      discriminator_name: str = 'AnimeDiscriminator',
      pretrained_name: str = 'VGGCaffePreTrained',
      **kwargs
  ):
    super().__init__()
    self.save_hyperparameters()

    # networks
    self.generator = self.GeneratorDict[generator_name]()
    self.pre_trained_ckpt = pre_trained_ckpt
    self.discriminator = self.DiscriminatorDict[discriminator_name]()
    self.lsgan_loss = LSGanLoss()
    self.pretrained = self.PreTrainedDict[pretrained_name]()
    self.l1_loss = nn.L1Loss()
    self.huber_loss = nn.SmoothL1Loss()

  def setup(self, stage: str):
    if stage == 'fit':
      if self.pre_trained_ckpt:
        ckpt = torch.load(self.pre_trained_ckpt)
        generatordict = dict(filter(lambda k: 'generator' in k[0], ckpt['state_dict'].items()))
        generatordict = {k.split('.', 1)[1]: v for k, v in generatordict.items()}
        self.generator.load_state_dict(generatordict, True)
        del ckpt
        del generatordict
        print("Success load pretrained generator from", self.pre_trained_ckpt)

    elif stage == 'test':
      pass

  def on_fit_start(self) -> None:
    self.pretrained.setup(self.device)

  def forward(self, im):
    return self.generator(im)

  def gram(self, x):
    b, c, h, w = x.shape
    gram = torch.einsum('bchw,bdhw->bcd', x, x)
    return gram / (c * h * w)

  def style_loss(self, style, fake):
    return self.l1_loss(self.gram(style), self.gram(fake))

  def con_sty_loss(self, real, anime, fake):
    real_feature_map = self.pretrained(real)
    fake_feature_map = self.pretrained(fake)
    anime_feature_map = self.pretrained(anime)

    c_loss = self.l1_loss(real_feature_map, fake_feature_map)
    s_loss = self.style_loss(anime_feature_map, fake_feature_map)

    return c_loss, s_loss

  def color_loss(self, con, fake):
    con = rgb2yuv(denormalize(con))
    fake = rgb2yuv(denormalize(fake))
    return (self.l1_loss(con[..., 0], fake[..., 0]) +
            self.huber_loss(con[..., 1], fake[..., 1]) +
            self.huber_loss(con[..., 2], fake[..., 2]))

  def discriminator_loss(self, real, gray, fake, real_blur):
    real_loss = torch.mean(torch.square(real - 1.0))
    gray_loss = torch.mean(torch.square(gray))
    fake_loss = torch.mean(torch.square(fake))
    real_blur_loss = torch.mean(torch.square(real_blur))
    return 1.2 * real_loss, 1.2 * gray_loss, 1.2 * fake_loss, 0.8 * real_blur_loss

  def generator_loss(self, fake_logit):
    return self.lsgan_loss._g_loss(fake_logit)

  def training_step(self, batch, batch_idx, optimizer_idx):
    input_photo, (input_cartoon, anime_gray_data), anime_smooth_gray_data = batch

    generated = self.generator(input_photo)
    generated_logit = self.discriminator(generated)

    if optimizer_idx == 0:  # train discriminator
      anime_logit = self.discriminator(input_cartoon)
      anime_gray_logit = self.discriminator(anime_gray_data)
      smooth_logit = self.discriminator(anime_smooth_gray_data)
      (d_real_loss, d_gray_loss,
       d_fake_loss, d_real_blur_loss) = self.discriminator_loss(
          anime_logit, anime_gray_logit,
          generated_logit, smooth_logit)

      d_loss_total = (self.hparams.d_adv_weight * (d_real_loss +
                                                   d_fake_loss +
                                                   d_gray_loss +
                                                   d_real_blur_loss))
      self.log_dict({'dis/d_loss': d_loss_total,
                     'dis/d_real_loss': d_real_loss,
                     'dis/d_fake_loss': d_fake_loss,
                     'dis/d_gray_loss': d_gray_loss,
                     'dis/d_real_blur_loss': d_real_blur_loss, })
      return d_loss_total
    elif optimizer_idx == 1:  # train generator
      c_loss, s_loss = self.con_sty_loss(input_photo, anime_gray_data, generated)
      c_loss = self.hparams.con_weight * c_loss
      s_loss = self.hparams.sty_weight * s_loss
      col_loss = self.color_loss(input_photo, generated) * self.hparams.color_weight
      g_loss = (self.hparams.g_adv_weight * self.generator_loss(generated_logit))
      g_loss_total = c_loss + s_loss + col_loss + g_loss
      self.log_dict({'gen/c_loss': c_loss,
                     'gen/s_loss': s_loss,
                     'gen/col_loss': col_loss,
                     'gen/g_loss': g_loss})
      return g_loss_total

  def configure_optimizers(self):
    opt_d = torch.optim.Adam(self.discriminator.parameters(),
                             lr=self.hparams.lr_d, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(self.generator.parameters(),
                             lr=self.hparams.lr_g, betas=(0.5, 0.999))
    return opt_d, opt_g

  def validation_step(self, batch, batch_idx):
    input_photo = batch
    log_images(self, {'input/real': input_photo,
                      'generate/anime': self.generator(input_photo)})

def infer_fn(model, args_list: List[str]):
  from datamodules.dsfunction import imread, denormalize, reduce_to_scale
  import datamodules.dstransform as transforms
  from utils.video import get_read_stream, get_writer_stream
  from pathlib import Path
  import cv2
  from utils.terminfo import INFO
  from more_itertools import chunked
  import argparse
  from tqdm import tqdm
  parse = argparse.ArgumentParser()
  parse.add_argument('--image_path',
                     help='input path, can be image,vedio,directory',
                     type=str)
  parse.add_argument('--device',
                     help='infer device',
                     type=str,
                     default='cpu', choices=['cpu', 'cuda'])
  parse.add_argument('--batch_size',
                     help='infer batch size',
                     type=int,
                     default=16)
  args = parse.parse_args(args_list)

  device = torch.device(args.device)

  resize2scale = transforms.ResizeToScale((256, 256), 32)
  infer_transform = transforms.Compose([
      resize2scale,
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
  ])
  model.setup('test')
  model.eval()

  model = model.to(device)

  def infer_batch(feed_im: torch.Tensor):
    feed_im = feed_im.to(device)
    out_im = model.forward(feed_im)
    draw_im = (denormalize(out_im.permute((0, 2, 3, 1)).
                           detach().
                           to('cpu').
                           numpy()) * 255).astype('uint8')
    return draw_im

  def infer_one_image(image_path: Path, output_root: Path):
    im = imread(image_path.as_posix())
    feed_im = infer_transform(im)
    draw_im = infer_batch(feed_im[None, ...])
    draw_im = draw_im[0]
    output_path = output_root / (image_path.stem + '_out' + image_path.suffix)
    cv2.imwrite(output_path.as_posix(), cv2.cvtColor(draw_im, cv2.COLOR_RGB2BGR))
    print(INFO, 'Convert', image_path, 'to', output_path)

  def infer_video(video_path: Path, output_root: Path, batch_size=16):
    read_stream, length, fps, height, width = get_read_stream(video_path)
    height, width = reduce_to_scale([height, width], resize2scale.size[::-1], resize2scale.scale)
    output_path = output_root / (image_path.stem + '_out' + '.mp4')
    writer_stream = get_writer_stream(output_path, fps, height, width)
    for frames in tqdm(chunked(read_stream, batch_size), total=length // batch_size):
      feed_im = torch.stack(
          [infer_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
           for frame in frames])
      draw_im = infer_batch(feed_im)
      for im in draw_im:
        writer_stream.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    writer_stream.release()

  image_path = Path(args.image_path)
  if image_path.is_file():
    output_root = image_path.parent
    if image_path.suffix in ['.mp4', '.flv']:
      infer_video(image_path, output_root, args.batch_size)
    else:
      infer_one_image(image_path, output_root)
  elif image_path.is_dir():
    output_root: Path = image_path.parent / (image_path.name + '_out')
    if not output_root.exists():
      output_root.mkdir()
    for p in image_path.iterdir():
      infer_one_image(p, output_root)

if __name__ == "__main__":
  run_common(AnimeGAN, AnimeGANDataModule, infer_fn)
