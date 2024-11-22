import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

class Modelo():
    def __init__(self):
        pass

    def CarregarDataset(self, path):
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)

    def TratamentoDeDados(self):
        self.x = self.df.drop('Species', axis=1)
        self.y = self.df['Species']
        pca = PCA(0.99)

        self.x = pca.fit_transform(self.x)

        print("N° de componentes do PCA: ")
        print(pca.n_components_)

    def Treinamento(self):
        self.x_train, self.x_teste, self.y_train, self.y_teste = train_test_split(self.x, self.y, test_size=0.3)

        self.model1 = RandomForestClassifier()
        self.model1.fit(self.x_train, self.y_train)

        self.model2 = SVC()
        self.model2.fit(self.x_train, self.y_train)
        

    def Teste(self):
        print("Score Modelo 1: ")
        print(self.model1.score(self.x_teste, self.y_teste))

        print("Score Modelo 2: ")
        print(self.model2.score(self.x_teste, self.y_teste))

    def Train(self):
        self.CarregarDataset("iris.data")  # Carrega o dataset especificado.

        # Tratamento de dados opcional, pode ser comentado se não for necessário
        self.TratamentoDeDados()

        self.Treinamento()  # Executa o treinamento do modelo

M = Modelo()
M.Train()
M.Teste()