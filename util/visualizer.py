'''
Partially modified.
Record loss and generate results.
'''

import numpy as np
import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

##
from PIL import Image
OUT_CHANNEL = 3
CROP_SIZE = 256

# colour map (from semantic segmentation visualization)
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
##

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.tf_log = opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                # Create an Image object
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                # Create a Summary value
                img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))

            # Create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.jpg' % (epoch, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.jpg' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.jpg' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()


    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    ''' Two-stage network outputs '''

    def combine_segs(self, seg_batch):
        combined_seg = seg_batch[:, :, :OUT_CHANNEL]
        for ind in range(1, 10):
            combined_seg += seg_batch[:, :, OUT_CHANNEL * ind: OUT_CHANNEL * (ind + 1)]
        return combined_seg

    # label to RGB image
    # segment: [20, CROP_SIZE, CROP_SIZE]
    def decode_label(self, segment):
        segment = segment.argmax(0) # [CROP_SIZE, CROP_SIZE]
        label_im = Image.new('RGB', (CROP_SIZE, CROP_SIZE))
        lable_pixels = label_im.load()

        for ix in range(CROP_SIZE):
            for jx in range(CROP_SIZE):
                lable_pixels[jx, ix] = label_colours[segment[ix, jx]]
        return label_im

    # trans_segs: [BATCH_SIZE, 30, H, W], float 32 [-1, 1]
    # ref_frames_foreground: [BATCH_SIZE, 3, H, W], float 32 [-1, 1]
    # ref_frames: [BATCH_SIZE, 3, H, W], float 32 [-1, 1]
    # gen_bi_mask: [BATCH_SIZE, 1, H, W], float 32 (0, 1)
    # comb_input: [BATCH_SIZE, 3, H, W], float 32 [-1, 1]
    # fake_image: [BATCH_SIZE, 3, H, W], float 32 [-1, 1]
    # comb_fake_image: [BATCH_SIZE, 3, H, W], float 32 [-1, 1]
    def save_image_batch(self, outputs_dict, inds, epoch_num, total_step, out_root=None):
        trans_batch = np.array(outputs_dict['trans_segs'])
        trans_batch = (np.transpose(trans_batch, (0, 2, 3, 1)) + 1) / 2.0 * 255.0

        ref_batch = np.array(outputs_dict['ref_frames_foreground'])
        ref_batch = (np.transpose(ref_batch, (0, 2, 3, 1)) + 1) / 2.0 * 255.0

        gen_msk_batch = np.array(outputs_dict['gen_bi_mask'])
        gen_msk_batch = np.transpose(gen_msk_batch, (0, 2, 3, 1)) * 255.0

        ref_msk_batch = np.array(outputs_dict['ref_bi_mask'])
        ref_msk_batch = np.transpose(ref_msk_batch, (0, 2, 3, 1)) * 255.0

        comb_in_batch = np.array(outputs_dict['comb_input'])
        comb_in_batch = (np.transpose(comb_in_batch, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        comb_in_batch = comb_in_batch[:, :, :, :3]

        out_batch = np.array(outputs_dict['fake_image'].detach())
        out_batch = (np.transpose(out_batch, (0, 2, 3, 1)) + 1) / 2.0 * 255.0

        comb_out_batch = np.array(outputs_dict['comb_fake_image'].detach())
        comb_out_batch = (np.transpose(comb_out_batch, (0, 2, 3, 1)) + 1) / 2.0 * 255.0

        final_batch = np.array(outputs_dict['ref_frames'])
        final_batch = (np.transpose(final_batch, (0, 2, 3, 1)) + 1) / 2.0 * 255.0

        if out_root is None:
            out_root = self.img_dir
        out_dir = '%s/epoch_%02d_%08d' % (out_root, epoch_num, total_step)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        trans_out_path = '{}/transformed_segments'.format(out_dir)
        ref_out_path = '{}/ref_frames'.format(out_dir)
        gen_out_path = '{}/generated_outputs'.format(out_dir)
        gen_msk_out_path = '{}/generated_masks'.format(out_dir)
        ref_msk_out_path = '{}/ref_masks'.format(out_dir)
        comb_in_path = '{}/comb_inputs'.format(out_dir)
        comb_out_path = '{}/comb_outputs'.format(out_dir)
        final_out_path = '{}/final_frames'.format(out_dir)

        if not os.path.exists(trans_out_path):
            os.mkdir(trans_out_path)
            os.mkdir(ref_out_path)
            os.mkdir(gen_out_path)
            os.mkdir(gen_msk_out_path)
            os.mkdir(ref_msk_out_path)
            os.mkdir(comb_in_path)
            os.mkdir(comb_out_path)
            os.mkdir(final_out_path)

        for cur_ind in range(len(inds)):
            trans_array = self.combine_segs(trans_batch[cur_ind])
            ref_array = ref_batch[cur_ind]
            out_array = out_batch[cur_ind]
            ##
            gen_msk_array = gen_msk_batch[cur_ind][:, :, 0]
            ref_msk_array = ref_msk_batch[cur_ind][:, :, 0]
            ##
            comb_in_array = comb_in_batch[cur_ind]
            comb_out_array = comb_out_batch[cur_ind]
            final_array = final_batch[cur_ind]

            trans_im = Image.fromarray(trans_array.astype(np.uint8))
            ref_im = Image.fromarray(ref_array.astype(np.uint8))
            out_im = Image.fromarray(out_array.astype(np.uint8))
            ##
            gen_msk_im = Image.fromarray(gen_msk_array.astype(np.uint8))
            ref_msk_im = Image.fromarray(ref_msk_array.astype(np.uint8))
            ##
            comb_in_im = Image.fromarray(comb_in_array.astype(np.uint8))
            comb_im = Image.fromarray(comb_out_array.astype(np.uint8))
            final_im = Image.fromarray(final_array.astype(np.uint8))

            trans_im.save('%s/result_%08d.png' % (trans_out_path, inds[cur_ind]))
            ref_im.save('%s/result_%08d.png' % (ref_out_path, inds[cur_ind]))
            out_im.save('%s/result_%08d.png' % (gen_out_path, inds[cur_ind]))
            ##
            gen_msk_im.save('%s/result_%08d.png' % (gen_msk_out_path, inds[cur_ind]))
            ref_msk_im.save('%s/result_%08d.png' % (ref_msk_out_path, inds[cur_ind]))
            ##
            comb_in_im.save('%s/result_%08d.png' % (comb_in_path, inds[cur_ind]))
            comb_im.save('%s/result_%08d.png' % (comb_out_path, inds[cur_ind]))
            final_im.save('%s/result_%08d.png' % (final_out_path, inds[cur_ind]))
