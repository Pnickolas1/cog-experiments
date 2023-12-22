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

# Working versions

### cog-experiments-v2

```
- 28beee53afff7ee070c72c2984dbabadea81e36d86fe8120acc85e0d4c56112b

```

### cog-experiments

```
  - 5e350f6e03a5efdf18084f9ec17497ab117c76f631c672a85ee21c0f2ba7bb50
```
