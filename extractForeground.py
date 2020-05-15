from PIL import Image, ImageChops
import numpy as np

inFile = "test_data/test_images/dog.JPG"
maskFile = "test_data/u2net_results/dog.png"

inImg = Image.open(inFile).convert('RGBA')
maskImg = Image.open(maskFile).convert('F')

inShape = np.array(inImg).shape

maskNp = np.array(maskImg)

newMask = np.zeros(inShape)
newMask[:,:,3] = maskNp

newMask = Image.fromarray(newMask.astype(np.uint8))

res = Image.fromarray(np.zeros(inShape).astype(np.uint8))
res.paste(inImg, mask=newMask)

res.save("res.png")

print(np.array(inImg).shape)
