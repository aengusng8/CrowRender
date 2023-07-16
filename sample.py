import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "mps"


def sample(datum, text):
    results = {}
    # Step 1: Create a context image
    context_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base"
    ).to(DEVICE)
    context_img = context_pipe(text).images[0]
    results["step_1"] = dict(context_img=context_img)

    # Step 2: Inpaint N bounding boxes
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting"
    ).to(DEVICE)
    results["step_2"] = iterative_inpainting(
        inpaint_pipe, datum, text, results["context_image"]
    )

    # (Optional) Step 3: Outpaint the above image
    # TODO: Implement this
    # mask_img = results["step_2"]["outpainting_mask_img"]
    # mask_imgs.append(mask_img)
    # generated_image = pipe(
    #     prompt,
    #     context_img,
    #     mask_img,
    #     guidance_scale=guidance_scale).images[0]

    # generated_images.append(generated_image)

    return results


def iterative_inpainting(
    context_img, pipe, datum, paste=False, guidance_scale=4.0, size=512
):
    d = datum
    d["unnormalized_boxes"] = d["boxes_unnormalized"]
    n_total_boxes = len(d["unnormalized_boxes"])

    context_imgs = []
    mask_imgs = []
    generated_images = []
    prompts = []
    notated_generated_images = []
    notated_context_images = []

    background_mask_img = Image.new("L", (size, size))
    background_mask_draw = ImageDraw.Draw(background_mask_img)
    background_mask_draw.rectangle(
        [(0, 0), background_mask_img.size], fill=255
    )  # fill the background with white

    for i in range(n_total_boxes):
        print("Iter: ", i + 1, "total: ", n_total_boxes)

        target_caption = d["box_captions"][i]

        mask_img = Image.new("L", context_img.size)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.rectangle([(0, 0), mask_img.size], fill=0)

        box = d["unnormalized_boxes"][i]
        if type(box) == list:
            box = torch.tensor(box)
        mask_draw.rectangle(box.long().tolist(), fill=255)  # fill the box with white
        background_mask_draw.rectangle(
            box.long().tolist(), fill=0
        )  # fill the box with black

        mask_imgs.append(mask_img.copy())

        prompt = f"{target_caption}"
        print(f'Inpainting "{target_caption}"')
        prompts += [prompt]
        context_imgs.append(context_img.copy())

        notated_context_img = context_img.copy()
        notated_context_img_draw = ImageDraw.Draw(notated_context_img)
        notated_context_img_draw.rectangle(box.long().tolist(), outline="red", width=5)
        notated_context_images.append(notated_context_img)

        generated_image = pipe(
            prompt, context_img, mask_img, guidance_scale=guidance_scale
        ).images[0]

        if paste:
            # context_img.paste(generated_image.crop(box.long().tolist()), box.long().tolist())

            src_box = box.long().tolist()

            # x1 -> x1 + 1
            # y1 -> y1 + 1
            paste_box = box.long().tolist()
            paste_box[0] -= 1
            paste_box[1] -= 1
            paste_box[2] += 1
            paste_box[3] += 1

            box_w = paste_box[2] - paste_box[0]
            box_h = paste_box[3] - paste_box[1]

            context_img.paste(
                generated_image.crop(src_box).resize((box_w, box_h)), paste_box
            )
        else:
            context_img = generated_image
        generated_images.append(context_img.copy())
        notated_generated_image = context_img.copy()
        notated_generated_image_draw = ImageDraw.Draw(notated_generated_image)
        notated_generated_image_draw.rectangle(
            box.long().tolist(), outline="red", width=5
        )
        notated_generated_images.append(notated_generated_image)

    outpainting_mask_img = background_mask_img

    return {
        "context_imgs": context_imgs,
        "mask_imgs": mask_imgs,
        "prompts": prompts,
        "generated_images": generated_images,
        "notated_generated_images": notated_generated_images,
        "outpainting_mask_img": outpainting_mask_img,
        "notated_context_images": notated_context_images,
    }
