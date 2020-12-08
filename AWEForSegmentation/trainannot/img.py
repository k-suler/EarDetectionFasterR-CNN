from PIL import Image

im = Image.open('0014.png')
im = im.convert('RGB')
r, g, b = im.split()
r = r.point(lambda i: i * 255)
out = Image.merge('RGB', (r, g, b))
out.show()