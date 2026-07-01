import torch

from torchvision import transforms
from torchvision.utils import save_image, flow_to_image

from loss.Multiscaleloss import multiscaleEPE
from testLoss.Loss2023 import sequence_loss
from util import flow_viz
from util.util import tensor2im, tensor2imFlo
from .base_model import BaseModel
from testLoss.Loss2022 import sequence_loss2022
from . import networks
from PIL import Image
from util.frame_utils import *
from loss import Multiscaleloss

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or testLoss phase. You can use this flag to add training-specific or testLoss-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/testLoss scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/testLoss scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A','real_C','fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/testLoss scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during testLoss time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        # 6 3 64 unet_128 instance x  normal 0.02 0
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        # # self.attn = AttentionLePE(dim=3)
        # self.attn = VisionAttention(
        #     dim=3,  # embedding dimension
        #     dim_index=1,  # where is the embedding dimension
        #     dim_heads=32,  # dimension of each head. defaults to dim // heads if not supplied
        #     heads=3,  # number of heads for multi-head attention
        #     depth=2,  # number of axial dimensions (images is 2, video is 3, or more)
        # )
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.MSELoss()
            self.criterionL2 = sequence_loss2022
            ###sself.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.valid = input['valid'].to(self.device)
        # print("setInput",self.valid.shape)
        self.real_C = input['C'].to(self.device)
        self.real_flo = input['flo'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.image_pathsc = input['C_paths']
        self.flo_path = input['flo_path']
        self.real_B = self.real_flo
        print("real_flo大小", self.real_flo.shape)
        print("real_flo大小", self.real_B.shape)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <testLoss>."""
        #cat A4c and B3c########################################
        # four_channel_img = self.real_A.convert('RGBA')
        # four_channel_img = four_channel_img.convert('RGB')
        # self.fake_B = self.netG(torch.cat([four_channel_img, self.real_C], dim=1))  # G(A)
        ####################################################### 1 6 265 1024
        self.fake_B = self.netG(torch.cat([self.real_A,self.real_C],dim=1))  # G(A)
        print("fake_B大小",self.fake_B.shape)
        # 1 2 256 1024
        img = self.fake_B.clone()
        # img = img[0]
        # save_image(img, 'img1.png')
        # 1 2 256 1024
        # img = flow_to_image(img)

        # 2 256 1024
        # _pflow = img[0].data.cpu().numpy().transpose(1, 2, 0)*255.0
        # 256 1024 2
        # 1 2 256 1024


        # print(_pflow.shape)
        # writeFlow("test.flo",_pflow)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((torch.cat([self.real_A,self.real_C],dim=1), self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        print("fakeAb",fake_AB.shape)
        pred_fake = self.netD(fake_AB.detach())
        # 对判别器，生成的假的AB结果给到判别器，希望判定为0
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((torch.cat([self.real_A,self.real_C],dim=1), self.real_B), 1)
        print("fakeAb1", fake_AB.shape)
        pred_real = self.netD(real_AB)
        # 对判别器，真实的结果给到判别器，希望判定为1
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients 总的判别器损失
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((torch.cat([self.real_A,self.real_C],dim=1),self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        # 对生成器，希望将生成的假的结果输入到鉴别器，鉴别结果为1
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G
        # (A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D 鉴别器不需要梯度
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights 更新模型参数
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
