{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import random\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import math\n",
    "import pickle\n",
    "import os\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "def perceptron(z):\n",
    "    return -1 if z<=0 else 1\n",
    "\n",
    "def ploss(yhat, y):\n",
    "    return max(0, -yhat*y)\n",
    "\n",
    "def ppredict(self, x):\n",
    "    return self(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Sonar_Model:\n",
    "    \n",
    "    def ppredict(self, x): \n",
    "        return\n",
    "\n",
    "    def __init__(self, dimension=None, weights=None, bias=None, activation=(lambda x: x), predict=(lambda x: x)):\n",
    "    \n",
    "        self._dim = dimension\n",
    "        self.w = weights or np.random.normal(size=self._dim)\n",
    "        self.w = np.array(self.w)\n",
    "        self.b = bias if bias is not None else np.random.normal()\n",
    "        self._a = activation\n",
    "        self.predict = predict.__get__(self)\n",
    "    \n",
    "    def __str__(self):\n",
    "        \n",
    "        return \"Simple cell neuron\\n\\\n",
    "        \\tInput dimension: %d\\n\\\n",
    "        \\tBias: %f\\n\\\n",
    "        \\tWeights: %s\\n\\\n",
    "        \\tActivation: %s\" % (self._dim, self.b, self.w, self._a.__name__)\n",
    "        return info\n",
    "    \n",
    "\n",
    "    def __call__(self, x):\n",
    "        yhat = self._a(np.dot(self.w, np.array(x)) + self.b)\n",
    "        return yhat\n",
    "    \n",
    "    def load_model(self, file_path):\n",
    "        '''\n",
    "        open the pickle file and update the model's parameters\n",
    "        '''\n",
    "        pass\n",
    "        \n",
    "    def save_model(self):\n",
    "        '''save your model as 'sonar_model.pkl' in the local path''' \n",
    "       # Data to be saved\n",
    "        pickle_out = open(\"save_n\",\"wb\")\n",
    "        pickle.dump(model, pickle_out)\n",
    "        pickle_out.close()\n",
    "     "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Sonar_Trainer:\n",
    "\n",
    "    def __init__(self, dataset, model):\n",
    "        \n",
    "        self.dataset = dataset\n",
    "        self.model = model\n",
    "        self.loss = ploss\n",
    "        \n",
    "    def cost(self, data):\n",
    "        \n",
    "        return np.mean([self.loss(self.model.predict(x), y) for x, y in data])\n",
    "    \n",
    "    def accuracy(self, data):\n",
    "        \n",
    "        return 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in data])\n",
    "    \n",
    "    def train(self, lr, ne):\n",
    "        \n",
    "        print(\"training model on data...\")\n",
    "        accuracy = self.accuracy(self.dataset)\n",
    "        print(\"initial accuracy: %.3f\" % (accuracy))\n",
    "        \n",
    "        for epoch in range(ne):\n",
    "            for d in self.dataset:\n",
    "                x, y = d\n",
    "                x = np.array(x)\n",
    "                yhat = self.model(x)\n",
    "                error = y - yhat\n",
    "                self.model.w += lr*(y-yhat)*x\n",
    "                self.model.b += lr*(y-yhat)\n",
    "            accuracy = self.accuracy(self.dataset)\n",
    "            #print('>epoch=%d, learning_rate=%.3f, accuracy=%.3f' % (epoch+1, lr, accuracy))\n",
    "            \n",
    "        print(\"training complete\")\n",
    "        print(\"final accuracy: %.3f\" % (self.accuracy(self.dataset)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "training model on data...\n",
      "initial accuracy: 53.365\n",
      "training complete\n",
      "final accuracy: 79.808\n"
     ]
    }
   ],
   "source": [
    "class Sonar_Data:\n",
    "\n",
    "    def __init__(self, file):\n",
    "        self.file = file\n",
    "        \n",
    "    def load(self):\n",
    "        with open(self.file, mode='rb') as f:\n",
    "            sonar_data = pickle.load(f)\n",
    "        self.mine = sonar_data.get('m')\n",
    "        self.rock = sonar_data.get('r')\n",
    "\n",
    "def main():\n",
    "\n",
    "    data = Sonar_Data(\"/Users/hugo/Documents/ArtificialIntelligence/Assignment 1/keio2019aia/data/assignment1/sonar_data.pkl\") # plug in path \n",
    "    data.load()\n",
    "    os.chdir(\"/Users/hugo/Documents/ArtificialIntelligence/Assignment 1/keio2019aia/data/assignment1\") # plug in path\n",
    "    data = [(list(d), 1) for d in data.mine]+[(list(d), -1) for d in data.rock]\n",
    "    random.shuffle(data)\n",
    "    \n",
    "    \n",
    "    model = Sonar_Model(dimension=60, activation = perceptron, predict = ppredict)  # specify the necessary arguments\n",
    "    trainer = Sonar_Trainer(data, model)\n",
    "    trainer.train(0.01, 100) # experiment with learning rate and number of epochs\n",
    "    #save_model()\n",
    "    pickle_out = open(\"sonar_model.pkl\",\"wb\")\n",
    "    pickle.dump(model, pickle_out)\n",
    "    pickle_out.close()\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
