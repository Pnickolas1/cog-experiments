#### first download the pre-trained weights

```
cog run script/download-weights
```

#### then you can run predictions

```
cog predict -i input_images=@zeke.zip -i use_face_detection_instead=True
```

### Example

---

Output is a `trained_model.tar` file

#### Civit location of coloring book weights:

```
https://civitai.com/models/136348/coloringbookredmond-coloring-book-lora-for-sd-xl

```

#### base on this repo on replicate:

[realvisxl2-lora-training](https://replicate.com/lucataco/realvisxl2-lora-training?input=form&output=preview)

### cog login & push

```
cog login
```

#### cog push

```
cog push r8.im/pnickolas1/cog-experiments

```

##### More

[ZipLoRA PyTorch](https://github.com/mkshing/ziplora-pytorch)
