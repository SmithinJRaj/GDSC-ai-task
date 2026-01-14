from PIL import Image, ImageDraw

img = Image.new("L", (28, 28), color=0)
draw = ImageDraw.Draw(img)
draw.text((12, 8), "4", fill=255)
img.save("digit.png")