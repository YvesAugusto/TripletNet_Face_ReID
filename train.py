import joblib
def train(model, data, epochs=20):
    for e in range(epochs):
        model.train_step(data)
    joblib.dump(model, 'triplet_model')

def load_model(name):
    return joblib.load(name)

