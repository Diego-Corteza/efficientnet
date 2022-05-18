
# Albumentations Set up
class AlbumData(dataset):

    def __init__(self, x=x_train, y=y_train, transforms=None):
        super().__init__()

        self.x = x
        self.y = y
        self.transform = transforms

        if self.y is None:
            self.len = len(self.x)
        else:
            try:
                assert len(self.x) == len(self.y)
                self.len = len(self.x)
            except AssertionError:
                print(f" the size of x ({len(self.x)} is different from y ({len(self.y)})")

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if self.y is not None:
            image, label = self.x[item], self.y[item]
            image = np.expand_dims(image, -1).astype(np.uint8)
            label = torch.from_numpy(np.array(label)).type(torch.LongTensor)
        else:
            image, label = self.x[item], None

        if self.transform is not None:
            aug_img = self.transform(image=image)
            image = aug_img["image"].astype(np.uint8).reshape(28, 28, 1)

        image = transforms.ToTensor()(image)

        if self.y is None:
            return image
        else:
            return image, label


