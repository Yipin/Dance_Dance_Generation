'''
Derived from pix2pixHD_model.py
Modified:
- Framework of two-stage model (comb_* represent network structures of fusion stage)
- Relativistic LSGAN
- Gradient penalty
- Semantic segmentation and pose feature loss
'''

import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
##
import sys
from . import networks_sp

class TwoStageModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):

        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True, True,  # foreground losses
                 True, use_gan_feat_loss, use_vgg_loss, True, True, True, True)  # fusion net losses

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, d_gp, g_sp,
                        comb_g_gan, comb_g_gan_feat, comb_g_vgg, comb_d_real,
                        comb_d_fake, comb_d_gp, comb_g_sp):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake, d_gp, g_sp,
                                        comb_g_gan,comb_g_gan_feat,comb_g_vgg,comb_d_real,comb_d_fake,
                                        comb_d_gp, comb_g_sp),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        comb_input_nc = opt.comb_label_nc

        ##### define networks        
        # Generator network
        # 30 + 45 = 75
        netG_input_nc = input_nc
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        comb_netG_input_nc = comb_input_nc
        self.comb_netG = networks.define_G(comb_netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

            comb_netD_input_nc = comb_input_nc + opt.output_nc
            self.comb_netD = networks.define_D(comb_netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)


        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.comb_netG, 'comb_G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                self.load_network(self.comb_netD, 'comb_D', opt.which_epoch, pretrained_path)


        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            if not opt.no_sp_loss:
                self.criterionSP = networks_sp.SPLoss(self.gpu_ids)
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real','D_fake', 'D_GP', 'G_SP',
                                               'comb_G_GAN','comb_G_GAN_Feat','comb_G_VGG','comb_D_real','comb_D_fake',
                                               'comb_D_GP', 'comb_G_SP')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the foreground local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            ## optimizer comb_G
            if opt.niter_fix_global > 0:
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.comb_netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print('------------- Only training the combine local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.comb_netG.parameters())
            self.comb_optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            ## optimizer comb_D
            params = list(self.comb_netD.parameters())
            self.comb_optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


    def encode_input(self, trans_segs, ref_poses, background_image, ref_frames_foreground=None, ref_frames=None, infer=False):

        trans_segs = trans_segs.data.cuda()
        trans_segs = Variable(trans_segs)

        ref_poses = ref_poses.data.cuda()
        ref_poses = Variable(ref_poses)

        background_image = background_image.data.cuda()
        background_image = Variable(background_image)


        # for training
        if ref_frames_foreground is not None:
            ref_frames_foreground = Variable(ref_frames_foreground.data.cuda())

        if ref_frames is not None:
            ref_frames = Variable(ref_frames.data.cuda())

        return trans_segs, ref_poses, background_image, ref_frames_foreground, ref_frames

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def comb_discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.comb_netD.forward(fake_query)
        else:
            return self.comb_netD.forward(input_concat)

    ##
    def compute_gradient_penalty(self, input_concat, real_data, fake_data, not_comb=True):
        alpha = torch.rand(list(input_concat.size())[0], 1, 1, 1)
        alpha = alpha.expand(real_data.size()).cuda()

        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        if not_comb:
            pred_interpolates = self.netD.forward(torch.cat((input_concat, interpolates), dim=1))
        else:
            pred_interpolates = self.comb_netD.forward(torch.cat((input_concat, interpolates), dim=1))

        gradient_penalty = 0
        if isinstance(pred_interpolates[0], list):
            for cur_pred in pred_interpolates:
                gradients = torch.autograd.grad(outputs=cur_pred[-1], inputs=interpolates,
                                          grad_outputs=torch.ones(cur_pred[-1].size()).cuda(),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradient_penalty += ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        else:
            sys.exit('The output of D should be a list (or lists for multi-scale D)')

        gradient_penalty = (gradient_penalty / self.opt.num_D) * 10
        return gradient_penalty

    # frame [BATCH_SIZE, H, W, 3]
    def get_green_mask(self, frame):
        logic_r = np.expand_dims(np.less_equal(frame[:, :, :, 0], 15), axis=3)
        logic_g = np.expand_dims(np.greater_equal(frame[:, :, :, 1], 245), axis=3)
        logic_b = np.expand_dims(np.less_equal(frame[:, :, :, 2], 15), axis=3)
        green_mask = np.all(np.concatenate((logic_r, logic_g, logic_b), axis=3), axis=-1)
        green_mask = np.transpose(1 - np.expand_dims(green_mask.astype(np.float32), axis=3), (0, 3, 1, 2))
        return green_mask

    def forward(self, trans_segs, ref_poses, background_image, ref_frames_foreground, ref_frames, infer=False):
        # Encode Inputs
        trans_segs, ref_poses, background_image, ref_frame_foreground, ref_frame = \
            self.encode_input(trans_segs, ref_poses, background_image, ref_frames_foreground, ref_frames)

        #######################################
        # foreground synthesis losses
        #######################################
        # Fake Generation
        input_concat = torch.cat((trans_segs, ref_poses), dim=1)
        fake_image = self.netG.forward(input_concat)

        # Fake Detection
        pred_fake_pool = self.discriminate(input_concat, fake_image, use_pool=True)
        # Real Detection
        pred_real = self.discriminate(input_concat, ref_frames_foreground)
        # Fake / Real Detection for GAN loss
        pred_fake = self.netD.forward(torch.cat((input_concat, fake_image), dim=1))
        _pred_real = self.netD.forward(torch.cat((input_concat, ref_frames_foreground), dim=1))

        # Define relativistic LSGAN
        loss_D_fake = self.criterionGAN(pred_fake_pool, pred_real, False)
        loss_D_real = self.criterionGAN(pred_real, pred_fake_pool, True)
        loss_G_GAN = (self.criterionGAN(pred_fake, _pred_real, True) +
                      self.criterionGAN(_pred_real, pred_fake, False)) * 0.5

        # get gradient penalty
        loss_GP = self.compute_gradient_penalty(input_concat.detach(), ref_frames_foreground.detach(), fake_image.detach())

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake[i][j],
                                                          pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, ref_frames_foreground) * self.opt.lambda_feat

        # Seg loss and Pose loss
        loss_G_SP = 0
        if not self.opt.no_sp_loss:
            loss_G_SP = self.criterionSP(fake_image, ref_frames_foreground) * self.opt.lambda_sp

        # detach
        fake_image_detached = fake_image.detach()
        ref_frame_foreground_detached = ref_frames_foreground.detach()

        ''' compute foreground background binary mask'''
        # [-1, 1] -> [0, 1] -> [0, 255]
        fake_image_detached_array = ((np.array(fake_image_detached) + 1.0) / 2.0 * 255.0).astype(np.uint8)
        ref_frame_foreground_array = ((np.array(ref_frame_foreground_detached) + 1.0) / 2.0 * 255.0).astype(np.uint8)
        # [3, H, W] -> [H, W, 3]
        fake_image_detached_array = np.transpose(fake_image_detached_array, (0, 2, 3, 1))
        ref_frame_foreground_array = np.transpose(ref_frame_foreground_array, (0, 2, 3, 1))
        #
        gen_bi_mask_array = self.get_green_mask(fake_image_detached_array)
        gen_bi_mask = torch.from_numpy(gen_bi_mask_array).cuda()
        ref_bi_mask_array = self.get_green_mask(ref_frame_foreground_array)
        ref_bi_mask = torch.from_numpy(ref_bi_mask_array).cuda()

        #######################################
        # combine (fusion) network losses
        #######################################
        comb_input = fake_image_detached * gen_bi_mask + background_image * (1 - gen_bi_mask)

        # concate with pose indicator
        comb_input = torch.cat((comb_input, ref_poses), dim=1)

        # Comb fake generation
        comb_fake_image = self.comb_netG.forward(comb_input)

        # Comb fake Detection
        comb_pred_fake_pool = self.comb_discriminate(comb_input, comb_fake_image, use_pool=True)
        # Comb real Detection
        comb_pred_real = self.comb_discriminate(comb_input, ref_frames)
        # Comb fake Detection for comb GAN loss
        comb_pred_fake = self.comb_netD.forward(torch.cat((comb_input, comb_fake_image), dim=1))
        _comb_pred_real = self.comb_netD.forward(torch.cat((comb_input, ref_frames), dim=1))

        comb_loss_D_fake = self.criterionGAN(comb_pred_fake_pool, comb_pred_real, False)
        comb_loss_D_real = self.criterionGAN(comb_pred_real, comb_pred_fake_pool, True)
        comb_loss_G_GAN = (self.criterionGAN(comb_pred_fake, _comb_pred_real, True) +
                           self.criterionGAN(_comb_pred_real, comb_pred_fake, False)) * 0.5

        # get gradient penalty
        comb_loss_GP = self.compute_gradient_penalty(comb_input.detach(), ref_frames.detach(), comb_fake_image.detach(), False)

        # Comb GAN feature matching loss
        comb_loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(comb_pred_fake[i]) - 1):
                    comb_loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(comb_pred_fake[i][j],
                                                          comb_pred_real[i][j].detach()) * self.opt.lambda_feat

        # Comb VGG feature matching loss
        comb_loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            comb_loss_G_VGG = self.criterionVGG(comb_fake_image, ref_frames) * self.opt.lambda_feat

        # Comb Seg loss and Pose loss
        comb_loss_G_SP = 0
        if not self.opt.no_sp_loss:
            comb_loss_G_SP = self.criterionSP(comb_fake_image, ref_frames) * self.opt.lambda_sp

        ''' build outputs dict '''
        outputs_dict = {'trans_segs': trans_segs,
                        'ref_frames_foreground': ref_frames_foreground,
                        'ref_frames': ref_frames,
                        'gen_bi_mask': gen_bi_mask,
                        'ref_bi_mask': ref_bi_mask,
                        'comb_input': comb_input,
                        'fake_image': fake_image,
                        'comb_fake_image': comb_fake_image}
        
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_GP, loss_G_SP,
                                  comb_loss_G_GAN, comb_loss_G_GAN_Feat, comb_loss_G_VGG, comb_loss_D_real,
                                  comb_loss_D_fake, comb_loss_GP, comb_loss_G_SP),
                 None if not infer else outputs_dict]

    def inference(self, trans_segs, ref_poses, background_image, ref_frames_foreground, ref_frames):
        # Encode Inputs        
        trans_segs, ref_poses, background_image, ref_frame_foreground, ref_frame = \
            self.encode_input(Variable(trans_segs), Variable(ref_poses), Variable(background_image),
                              Variable(ref_frames_foreground), Variable(ref_frames), infer=True)

        # Fake Generation
        input_concat = torch.cat((trans_segs, ref_poses), dim=1)
           
        if torch.__version__.startswith('0.4') or torch.__version__.startswith('1'):
            with torch.no_grad():

                #######################################
                # foreground synthesis
                #######################################
                fake_image = self.netG.forward(input_concat)

                # detach
                fake_image_detached = fake_image.detach()
                ref_frame_foreground_detached = ref_frames_foreground.detach()

                ''' compute foreground background binary mask'''
                # [-1, 1] -> [0, 1] -> [0, 255]
                fake_image_detached_array = ((np.array(fake_image_detached) + 1.0) / 2.0 * 255.0).astype(np.uint8)
                ref_frame_foreground_array = ((np.array(ref_frame_foreground_detached) + 1.0) / 2.0 * 255.0).astype(
                    np.uint8)
                # [3, H, W] -> [H, W, 3]
                fake_image_detached_array = np.transpose(fake_image_detached_array, (0, 2, 3, 1))
                ref_frame_foreground_array = np.transpose(ref_frame_foreground_array, (0, 2, 3, 1))
                #
                gen_bi_mask_array = self.get_green_mask(fake_image_detached_array)
                gen_bi_mask = torch.from_numpy(gen_bi_mask_array).cuda()
                ref_bi_mask_array = self.get_green_mask(ref_frame_foreground_array)
                ref_bi_mask = torch.from_numpy(ref_bi_mask_array).cuda()

                #######################################
                # combine (fusion) synthesis
                #######################################
                comb_input = fake_image_detached * gen_bi_mask + background_image * (1 - gen_bi_mask)

                # concate with pose indicator
                comb_input = torch.cat((comb_input, ref_poses), dim=1)

                # Comb fake generation
                comb_fake_image = self.comb_netG.forward(comb_input)

                ''' build outputs dict '''
                outputs_dict = {'trans_segs': trans_segs,
                                'ref_frames_foreground': ref_frames_foreground,
                                'ref_frames': ref_frames,
                                'gen_bi_mask': gen_bi_mask,
                                'ref_bi_mask': ref_bi_mask,
                                'comb_input': comb_input,
                                'fake_image': fake_image,
                                'comb_fake_image': comb_fake_image}
        else:
            sys.exit('torch version is supposed to be equal to or higher than 0.4.')
        return outputs_dict

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        ##
        self.save_network(self.comb_netG, 'comb_G', which_epoch, self.gpu_ids)
        self.save_network(self.comb_netD, 'comb_D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(TwoStageModel):
    def forward(self, inp):
        trans_segs, ref_poses, background_image, ref_frames_foreground, ref_frames = inp
        return self.inference(trans_segs, ref_poses, background_image, ref_frames_foreground, ref_frames)

        
