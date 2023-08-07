from PIL import Image

width = 2560
height = 1360

im = Image.new(mode="RGB", size=(width, height), color=(255,255,255))


t0 = Image.open("0.png")
t1 = Image.open("1.png")
t2 = Image.open("2.png")
t3 = Image.open("3.png")

im.paste(t0, (50,50))
im.paste(t1, (width-250,50))
im.paste(t2, (50,height-250))
im.paste(t3, (width-250,height-250))

im.save("calibrator.png")


