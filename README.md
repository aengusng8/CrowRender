# CrowdRender
Generate image from layout with multiple bounding boxes (e.g., 20-100 boxes)
- *Input*: N x categories (optional: and a set of their bounding boxes)
- *Output*: An image with N x objects at the specified locations
- Constraint: use pretrained DMs only

## Methodology 1 (Very few times it works!)
**Updated 1:** It works! More tuning hyperparas (e.g., `guidance_scale`) probably increases qualitative and quantitative metrics of layout-to-image task.
**Updated 2:** It failed too! The behavior of the inpainting model is strange; it sometimes inpaints the object but also erases the right object.

**Step 1: Create a background image**

**Step 2: Iteratively inpaint N bounding boxes**

**(Optional) Step 3: Outpaint the above image**Consstant


## Methodology 2 (Failed)
Updated: This method is failed, because the diffusion model at step 2 doesn't generate a great image as a strong prior for another diffusion model at step 3.

**(Optional) Step 1: Synthesize a proper layout**
- *Input*: N x categories (optional: and a set of their bounding boxes)
- *Output*: Set of N x bounding boxes with a plausible arrangement
- *Purpose*: Instead of manually writing out 20-100 bounding boxes, this step aims to synthesize a plausible arrangement of element bounding boxes, considering optional constraints such as the type or position of a specific element.
- *Usage*: a layout generator (e.g. LayoutDM) with `C->P+S` task or `refinement` task
  
**Step 2: Synthesize a image with K (=9) bounding boxes at boudary**
  - *Input*: K x bounding boxes (a subset of N x bounding boxes)
  - *Output*: An image containing K x objects with plausible context/background
  - *Purpose*: To provide a background image for an inpainting model, this step generates K bounding boxes placed at the boundary of the image. The generated context and background make the inpainting process more effective.
  - *Usage*: a layout-to-image generator (e.g. LayoutDiffusion)
  
**Step 3: Iteratively inpaint (N - K) bounding boxes**
  - *Input*: An image containing K x objects with plausible context/background
  - *Output*: An image with N x objects at the specified locations
  - *Purpose*: Inpaint the remaining (N - K) bounding boxes into the image in an iterative manner. This step ensures the completion of the overall image by filling in the missing elements within the plausible context and background.
  - *Usage*: a inpainting model (e.g. Stable Diffusion)

