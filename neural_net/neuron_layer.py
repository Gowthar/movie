import neuron as Neuron

class NeuronLayer:
    def __init__(self, n_neurons, n_inputs):
        self.n_neurons = n_neurons
        self.neurons = [Neuron( n_inputs )
                        for _ in range(0,self.n_neurons)]

    def __str__(self):
        return 'Layer:\n\t'+'\n\t'.join([str(neuron)
                                         for neuron in self.neurons])+''
