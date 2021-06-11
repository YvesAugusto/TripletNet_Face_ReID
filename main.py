from tools import *
from triplet import TripletModel
from train import *
from data import *
import argparse


if '__name__' == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trs", help="train a model and saves it")
    parser.add_argument("--p", help="loads a trained model and makes prediction")

    data = load_data()
    args = parser.parse_args()

    if args.trs:
        model = create_model()
        triplet_model = TripletModel(model)
        triplet_model.compile(loss = triplet_loss, optimizer = tf.keras.optimizers.Adam(0.0001))

        train(model = triplet_model, data = data, epochs = 20)

    elif args.p:
        triplet_model = load_model('triplet_model')
