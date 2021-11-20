import torch
from torch.autograd import Variable  # module reponsible to gradient decent
import cv2
# will do require transformations to fit the images in the NN
# VOC classes will do enconding
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio


# torch.cuda.is_available = lambda: False
torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def detect(image, network, transform):

    height, width = image.shape[:2]  # gets first 2 channels
    image_transform = transform(image)[0]  # we get first element

    # convertion to Torch Tensor
    # the permute its to adjust the indixes of color to GRb instead of RGB
    x = torch.from_numpy(image_transform).permute(2, 0, 1)
    # adding fake dimension. Always add to first dimentions to the batch
    x = x.unsqueeze(0)

     # feed image to NN
    with torch.no_grad():
        y = network(x)

    detections = y.data  # first attreibute is tensor and second is gradient
    # create a tensor object of dimensions [width, height, width, height]
    # this is because the position needs to be normalised between 0,1 and positions of the objects
    scale = torch.Tensor([width, height, width, height])
    """
    detections = [batch, num of classes (objects),number of occurance of class,(score,x0,y0,x1,y1)]
    """

    for i in range(detections.size(1)):
        j = 0  # occurtance tracking
        while detections[0, i, j, 0] >= 0.6:
            # We get the coordinates of the points at the upper left and the lower right of the detector rectangle.
            # we consider scale and convert to numpy array
            pt = (detections[0, i, j, 1:]*scale).numpy()

            cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(
                pt[2]), int(pt[3])), (0, 255, 0), 3)

            cv2.putText(
                image, labelmap[i-1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            j += 1

    return image


def main():
    # we will use the test because we will use as pretrained moodel
    network = build_ssd("test")
    network.load_state_dict(torch.load("ssd300_mAP_77.43_v2.pth",
                            map_location=lambda storage, loc: storage))  # load weights

    # scale on which the NN was trained
    transforms = BaseTransform(network.size, (104/256.0, 117/256.0, 123/256.0))

    video = imageio.get_reader("funny_dog.mp4")
    fps = video.get_meta_data()["fps"]
    writer = imageio.get_writer("output.mp4",fps=fps)
    for i ,image in enumerate(video):
        image = detect(image,network.eval(),transforms)
        writer.append_data(image)
        print(i)

    writer.close()

if __name__ == "__main__":
    main()
