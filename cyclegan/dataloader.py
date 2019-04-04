import os
import cv2
import pickle
import numpy as np


class dataloader:
    def __init__(self, root_path, domain_path_A, domain_path_B, bs=64, max_size=1000, if_generate=False):
        self.root_path = root_path
        self.max_size = max_size
        self.bs = bs
        self.pointerA = 0
        self.pointerB = 0

        try:
            os.makedirs(self.root_path)
        except OSError:
            pass

        pkl_A = os.path.join(self.root_path, domain_path_A.split("/")[-2] + ".pkl")
        pkl_B = os.path.join(self.root_path, domain_path_B.split("/")[-2] + ".pkl")

        if not os.path.exists(pkl_A) or if_generate:
            print(">>", "generate", domain_path_A)
            self.getImages(domain_path_A)
        if not os.path.exists(pkl_B) or if_generate:
            print(">>", "generate", domain_path_B)
            self.getImages(domain_path_B)

        self.domainA = pickle.load(open(pkl_A, "rb"))
        self.domainB = pickle.load(open(pkl_B, "rb"))

        print(">>", "shape of domain A", self.domainA.shape)
        print(">>", "shape of domain B", self.domainB.shape)

    def getImages(self, img_path):
        imgs = os.listdir(img_path)
        data = np.zeros((1, 128, 128, 3))
        for i, img in enumerate(imgs):
            if img[-3:] not in ['jpg', 'png']:
                continue
            if i >= self.max_size:
                break
            image = cv2.imread(img_path + "/" + img)
            shaped = cv2.resize(image, (128, 128))
            shaped = np.expand_dims(shaped, 0)
            data = np.concatenate((data, shaped), axis=0)

        pickle.dump(data[1:], open(os.path.join(self.root_path,
                                            img_path.split("/")[-2] + ".pkl"), "wb"))
        print(">>", "generate", os.path.join(self.root_path,
                                             img_path.split("/")[-2] + ".pkl"), "DONE")

    def next(self):
        if (self.pointerA + 1) * self.bs > len(self.domainA):
            self.pointerA = 0
        if (self.pointerB + 1) * self.bs > len(self.domainB):
            self.pointerB = 0

        imageA = self.domainA[self.pointerA * self.bs:(self.pointerA + 1) * self.bs]
        imageB = self.domainB[self.pointerA * self.bs:(self.pointerB + 1) * self.bs]

        imageA = imageA.transpose(0, 3, 1, 2)
        imageB = imageB.transpose(0, 3, 1, 2)

        self.pointerA += 1
        self.pointerB += 1
        return imageA, imageB

    def iters(self):
        return int(len(self.domainA)/self.bs)

    def test(self):
        imageA, imageB = self.next()
        for i in range(3):
            A = imageA[i].transpose(1, 2, 0)
            B = imageB[i].transpose(1, 2, 0)
            print(A.shape, B.shape)
            cv2.imwrite("./data/A_{}.jpg".format(i), A)
            cv2.imwrite("./data/B_{}.jpg".format(i), B)
            break


if __name__ == "__main__":
    a = dataloader("./data",
                   "/home/szk/PycharmProjects/open-reid/examples/data/viper/images",
                   "/home/szk/PycharmProjects/open-reid/examples/data/cuhk03/images",
                   if_generate=False)
    a.test()
