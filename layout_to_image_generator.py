"""
**Step 2: Synthesize a image with K (=9) bounding boxes at boudary**
  - *Input*: K x bounding boxes (a subset of N x bounding boxes)
  - *Output*: An image containing K x objects with plausible context/background
  - *Purpose*: To provide a background image for an inpainting model, this step generates K bounding boxes placed at the boundary of the image. The generated context and background make the inpainting process more effective.
  - *Usage*: a layout-to-image generator (e.g. LayoutDiffusion)
"""

import argparse
from omegaconf import OmegaConf

import torch
import torch as th

from repositories.LayoutDiffusion.scripts.launch_gradio_app import (
    layout_to_image_generation,
)
from repositories.LayoutDiffusion.layout_diffusion.layout_diffusion_unet import (
    build_model,
)
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


class LayoutToImageGenerator:
    def __init__(self, config_file, pretrained_model_path=None, device="cuda"):
        cfg = OmegaConf.load(config_file)
        if pretrained_model_path is not None:
            cfg.sample.pretrained_model_path = pretrained_model_path
        cfg.config_file = config_file
        cfg.sample.timestep_respacing[0] = "25"
        cfg.sample.sample_method = "dpm_solver"

        self.device = torch.device(device)
        self.cfg, self.model_fn, self.noise_schedule = self.init(cfg)

    def sample(self, bboxes):
        """
        Generate a image with K (=9) bounding boxes at boudary
        """
        return layout_to_image_generation(
            self.cfg, self.model_fn, self.noise_schedule, bboxes
        )

    @torch.no_grad()
    def init(self, cfg):
        print(OmegaConf.to_yaml(cfg))

        print("creating model...")
        model = build_model(cfg)
        model.to(self.device)
        print(model)

        if cfg.sample.pretrained_model_path:
            print("loading model from {}".format(cfg.sample.pretrained_model_path))
            checkpoint = torch.load(cfg.sample.pretrained_model_path, map_location="cpu")

            try:
                model.load_state_dict(checkpoint, strict=True)
                print('successfully load the entire model')
            except:
                print('not successfully load the entire model, try to load part of model')

                model.load_state_dict(checkpoint, strict=False)

        model.to(self.device)
        if cfg.sample.use_fp16:
            model.convert_to_fp16()
        model.eval()

        def model_fn(x, t, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, **kwargs):
            assert obj_class is not None
            assert obj_bbox is not None

            cond_image, cond_extra_outputs = model(
                x, t,
                obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
                is_valid_obj=is_valid_obj
            )
            cond_mean, cond_variance = th.chunk(cond_image, 2, dim=1)

            obj_class = th.ones_like(obj_class).fill_(model.layout_encoder.num_classes_for_layout_object - 1)
            obj_class[:, 0] = 0

            obj_bbox = th.zeros_like(obj_bbox)
            obj_bbox[:, 0] = th.FloatTensor([0, 0, 1, 1])

            is_valid_obj = th.zeros_like(obj_class)
            is_valid_obj[:, 0] = 1.0

            if obj_mask is not None:
                obj_mask = th.zeros_like(obj_mask)
                obj_mask[:, 0] = th.ones(obj_mask.shape[-2:])

            uncond_image, uncond_extra_outputs = model(
                x, t,
                obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
                is_valid_obj=is_valid_obj
            )
            uncond_mean, uncond_variance = th.chunk(uncond_image, 2, dim=1)

            mean = cond_mean + cfg.sample.classifier_free_scale * (cond_mean - uncond_mean)

            if cfg.sample.sample_method in ['ddpm', 'ddim']:
                return [th.cat([mean, cond_variance], dim=1), cond_extra_outputs]
            else:
                return mean

        print("creating diffusion...")

        noise_schedule = NoiseScheduleVP(schedule='linear')

        print('sample method = {}'.format(cfg.sample.sample_method))
        print("sampling...")

        return cfg, model_fn, noise_schedule


def systhesize_initial_image(bboxes):
    """
    Synthesize a image with K (=9) bounding boxes at boudary
    """
    pass


if __name__ == "__main__":

    def test_systhesize_initial_image():
        num_obj = 10

        obj_class = [
            "image",
            "train",
            "pavement",
            "sky-other",
            "tree",
            "wall-other",
            "pad",
            "pad",
            "pad",
            "pad",
        ]

        obj_bbox = [
            [0.0, 0.0, 1.0, 1.0],
            [
                0.09859374910593033,
                0.24045585095882416,
                0.9391249418258667,
                0.8741595149040222,
            ],
            [0.0, 0.49857550859451294, 1.0, 1.0],
            [0.0, 0.0, 0.515625, 0.5441595315933228],
            [0.28125, 0.0, 1.0, 0.41310539841651917],
            [0.8999999761581421, 0.3817663788795471, 1.0, 0.692307710647583],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]

        custom_layout_dict = {
            "obj_class": obj_class,
            "obj_bbox": obj_bbox,
            "num_obj": num_obj,
        }

        config_file = "/Users/ducanhnguyen/Desktop/deep_learning_projects/minLayout/configs/COCO-stuff_128x128/LayoutDiffusion_large.yaml"
        pretrained_model_path = (
            "/Users/ducanhnguyen/Desktop/deep_learning_projects/minLayout/pretrained_models/COCO-stuff_128x128_LayoutDiffusion_large_ema_0300000.pt"
        )

        L2I_generator = LayoutToImageGenerator(config_file, pretrained_model_path, device="mps") # TODO: device="cuda"
        L2I_generator.sample(custom_layout_dict)

    test_systhesize_initial_image()
