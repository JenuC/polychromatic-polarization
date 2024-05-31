import os
import glob
import pathlib
from skimage import img_as_uint, io
import tifffile as tf
import re



# Specify the path to your JSON file
tif_folder =  os.getcwd()


data_folder = pathlib.Path(tif_folder) #pathlib.Path(tif_folder).parent
tif_strip_folder = data_folder.joinpath("strip_metadata")
if not os.path.exists(tif_strip_folder):
    os.mkdir(tif_strip_folder)
    
first_file = sorted( glob.glob(os.path.join(tif_folder,'*.tif')), key = lambda x: os.path.getmtime(x))[0]

mmfile = tf.TiffFile(first_file)

metadata =  mmfile.micromanager_metadata

pixel_size = float(re.findall('PhysicalSizeX="(\d*.\d*)"',mmfile.ome_metadata)[0])

data = metadata

def get_position_from_device(position):
    #print(position)
    for device in position[
                    "DevicePositions"
                ]:
        if (device['Device']=='XYStage'):
            x_,y_ = device['Position_um']
            return([x_,y_])
            

pos = data["Summary"]["StagePositions"]
label_list = {}
position_list = {}
for ix, position in enumerate(pos):
    #print(ix, position)
    position_list.update(
        {
            (position["GridCol"], position["GridRow"]): get_position_from_device(position)
        }
    )
    label_list.update(
        {
            (position["GridCol"], position["GridRow"]): position[
                "Label"
            ]
        }
    )

print(len(position_list))



save_config = True
save_tif = True

fl = glob.glob(tif_folder + r"\*.tif")

if save_config:
    # position_list = np.array(position_list)
    #pixel_size = 1.105
    # pixel_size = 1
    # with open(os.path.join(r"D:\__Data\20230110_Mike_tissue_4x", 'TileConfiguration.txt'), 'w') as text_file:
    with open(
        os.path.join(tif_strip_folder, "TileConfiguration.txt"),
        "w",
    ) as text_file:
        print("dim = 2", file=text_file)
        for pos in range(len(fl)):
            fn = pathlib.Path(fl[pos]).name
            # print(fn)
            xy = re.findall(r"Pos(\d{3})_(\d{3})", fn)
            xy = [(int(x), int(y)) for x, y in xy][0]
            # print(xy)
            # print(position_list[xy])
            x, y = position_list[xy]
            # x = int(position_list[pos, 0] / pixel_size)
            # y = int(position_list[pos, 1] / pixel_size)
            # print(pos,fl[pos])
            # print(x,y,label_list[xy])
            x = x / pixel_size
            y = y / pixel_size

            # break
            # print(f"{fn}; ; ({int(x)}, {int(y)})")
            # print("====")
            print(f"{fn}; ; ({int(x)}, {int(y)})", file=text_file)
            # if pos==2:
            #    break
    text_file.close()
    

stitch_folder = tif_strip_folder

if save_tif:
    for fn in fl:
        img = tf.imread(fn)#.asarray()
        print(img.shape)
        nfn = os.path.join(stitch_folder, f"{pathlib.Path(fn).name}")
        io.imsave(nfn, img_as_uint(img), check_contrast=False)
