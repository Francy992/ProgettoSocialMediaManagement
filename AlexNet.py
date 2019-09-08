import torchvision.models as models
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from Functions import *
from torch.autograd import Variable
import os

np.random.seed(1234)
torch.random.manual_seed(1234)

class AlexNet:

    def __init__(self, train = False, epochs = 20, dataset_path = "Dataset/", pth="net_150.pth"):
        self.width = 256
        self.height = 256
        self.n_classes = 2
        self.epochs = epochs
        self.train = train
        self.neural_net = "alexnet"
        self.dataset_path = dataset_path
        self.pth = pth
        self.net = models.alexnet(pretrained=True)

        self.softmax = nn.Softmax(dim=1)

        self.mean_squeezenet =[0.485, 0.456, 0.406]
        self.std_squeezenet =[0.229, 0.224, 0.225]
        transformss = transforms.Compose([
                                      #transforms.Resize((width,height)),
                                      transforms.ToTensor(), 
                                      #transforms.Normalize(mean_pre_trained,std_pre_trained),
                                      transforms.Normalize(self.mean_squeezenet, self.std_squeezenet),
                                  ])
        self.dataset_train_squeezenet = ScenesDataset('','train.txt',transform=transformss)
        self.dataset_test_squeezenet = ScenesDataset('','test.txt',transform=transformss)
        train_loader_squeezenet = DataLoader(self.dataset_train_squeezenet, batch_size=20, num_workers=0, shuffle=True)
        test_loader_squeezenet = DataLoader(self.dataset_test_squeezenet, batch_size=20, num_workers=0)


        '''self.barilla_train = ScenesDataset(self.dataset_path + 'Train',self.dataset_path + 'train.txt',transform=self.transformss)
        self.barilla_test = ScenesDataset(self.dataset_path + 'Test',self.dataset_path + 'test.txt',transform=self.transformss)
        self.barilla_train_loader = torch.utils.data.DataLoader(self.barilla_train, batch_size=10, num_workers=0)
        self.barilla_test_loader = torch.utils.data.DataLoader(self.barilla_test, batch_size=10, num_workers=0)
		'''

    '''def evaluate(self):

        if self.neural_net == "alexnet":
            self.net = models.alexnet(pretrained=True)
        else:
            self.net = models.squeezenet1_0(pretrained=True)

        for param in self.net.parameters():
            param.requires_grad = False

        if self.neural_net == "alexnet":
            self.net.classifier[6] = nn.Linear(4096, self.n_classes) #Numero esatto di classi nel nostro dataset.
        else:
            self.net.classifier[1] = nn.Conv2d(512, self.n_classes, kernel_size=(1, 1), stride=(1, 1))

        print("parameters: ",sum([p.numel() for p in self.net.parameters()]))

        if self.train:
            #self.net.load_state_dict(torch.load("./net_crop.pth"))
            lenet_mnist, lenet_mnist_logs = train_classification(self.net, epochs=self.epochs, train_loader = self.barilla_train_loader,
                                                                 test_loader = self.barilla_test_loader, lr=0.001)
        else:
            self.net.load_state_dict(torch.load(self.dataset_path + self.pth))


        if self.train:
            lenet_mnist_predictions, lenet_mnist_gt = test_model_classification(self.net, self.barilla_train_loader)
            print ("Accuracy squeezenet1_0 di train: %0.2f" % accuracy_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1)))

            lenet_mnist_predictions, lenet_mnist_gt = test_model_classification(self.net, self.barilla_test_loader)
            print ("Accuracy squeezenet1_0 di test: %0.2f" % accuracy_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1)))


        if torch.cuda.is_available():
            self.net = self.net.cuda()
            torch.cuda.empty_cache()
        self.net.eval()

        barilla_train_loader_OB = torch.utils.data.DataLoader(self.barilla_train, batch_size=1, num_workers=0, shuffle=True)
        barilla_test_loader_OB = torch.utils.data.DataLoader(self.barilla_test, batch_size=1, num_workers=0)

        input_for_datafram_train, label_array_train = get_dataframe(barilla_train_loader_OB, self.net)
        df = pd.DataFrame(input_for_datafram_train)

        self.knn_1.fit(df, label_array_train)
        self.knn_3.fit(df, label_array_train)
        self.knn_5.fit(df, label_array_train)

        feature_test = extract_features(barilla_test_loader_OB, self.net)
        if self.train:
            print("Accuracy con rete preallenata e dataset base.")
            print("1NN accuracy: ", accuracy(self.knn_1, feature_test))
            print("3NN accuracy: ", accuracy(self.knn_3, feature_test))
            print("5NN accuracy: ", accuracy(self.knn_5, feature_test))
	'''
    def load_model(self, model):
        #self.net.classifier[6] = nn.Linear(4096, self.n_classes)
        self.net.load_state_dict(torch.load("./" + model + ".pth"))

    
    def get_net_class(self, img):
        self.net.eval()
        self.net.cpu()
        
        transform = transforms.Compose([
                                      #transforms.Resize((width,height)),
                                      transforms.ToTensor(), 
                                      #transforms.Normalize(mean_pre_trained,std_pre_trained),
                                      transforms.Normalize(self.mean_squeezenet, self.std_squeezenet),
                                  ])
        img = transform(img)
        x = Variable(img).cpu()
        x = x.unsqueeze(0)
        pred = self.softmax(self.net(x)).data.cpu().numpy().copy()
        pred = pred.argmax(1)

        filepath = 'classes.txt'
        with open(filepath, "r+") as fp:
            for i in range(self.n_classes):
                line = fp.readline()
                n, c = line.strip().split(",")

                if int(c) == int(pred):
                    print("name: ", n, " c: ", c)
                    name = n

        return name, pred

   

#an = AlexNet(train = True, epochs=150)
#an.neural_net = "net_squeez_150"
#an.evaluate()

# Alexnet 50 epoche:
#   1NN accuracy:  0.7857142857142857
#   3NN accuracy:  0.8214285714285714
#   5NN accuracy:  0.8571428571428571
#   5NN accuracy:  0.8571428571428571