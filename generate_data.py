import cv2 as cv
import os


def pyramid_demo(image, s, index, dataset):
    level = 3
    temp = image.copy()
    pyramid_images = []
    for i in range(0, level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        temp = dst.copy()
        if i == 0:
            cv.imwrite(dataset + '400_400/rgb/' + s[index], dst)
        if i == 1:
            cv.imwrite(dataset + '200_200/rgb/' + s[index], dst)
        if i == 2:
            cv.imwrite(dataset + '100_100/rgb/' + s[index], dst)

    return pyramid_images





if __name__ == "__main__":
    dataset = 'data/nsvf/Synthetic_NeRF/Chair'
    path = dataset + '/rgb'
    files = os.listdir(path)
    s = []
    for file_ in files:
        if not os.path.isdir(path + file_):
            f_name = str(file_)
            s.append(f_name)

    os.makedirs("{}100_100/rgb".format(dataset), exist_ok=False)
    os.makedirs("{}200_200/rgb".format(dataset), exist_ok=False)
    os.makedirs("{}400_400/rgb".format(dataset), exist_ok=False)
    os.makedirs("{}800_800/rgb".format(dataset), exist_ok=False)

    number = len(s)

    for i in range(0, number):
        src = cv.imread(dataset + '/rgb/' + s[i], -1)
        cv.imwrite(dataset + '800_800/rgb/' + s[i], src)
        pyramid_demo(src, s, i, dataset)








