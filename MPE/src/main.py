from data import yahoo_load, mpe_yahoo_load, yahoo_load_bpr
from trainer import RmfTrainer, FmfTrainer, BprTrainer
import argparse
from evaluator import Evaluator
import tensorflow as tf
import warnings


def main(args):
    if args.model == 'BPR':
        # BPR needs to sample (positive, negative) pair, we load the yahoo data as follows
        num_ratings, num_user, num_item, train, test, test_rare = yahoo_load_bpr()
        latent_dim = 30
        learning_rate = 0.005
        regu_lam = 0.0001
        iters = 300
        batch_size = 32
        Optimizer = tf.train.AdamOptimizer
        warnings.filterwarnings("ignore")
        bpr = BprTrainer(train, test, num_user, num_item,
                         latent_dim, Optimizer, learning_rate, regu_lam)
        bpr.build_model(test, iters, batch_size)
        evaluator = Evaluator(test=test, rare=False, name='bpr')
        evaluator.evaluate()
        evaluator_rare = Evaluator(test=test_rare, rare=True, name='bpr')  # evaluate the performance on rare items
        evaluator_rare.evaluate()

    if args.model == 'FMF':
        latent_dim = 200
        regu_lam = 0.00001
        iters = 300
        batch_size = 2**15
        learning_rate = 0.005
        weight = 1
        warnings.filterwarnings("ignore")
        train, test, test_rare, num_users, num_items = yahoo_load()
        trainer_fmf = FmfTrainer(latent_dim=latent_dim, regu_lam=regu_lam, iters=iters, batch_size=batch_size,
                                 learning_rate=learning_rate, num_users=15400, num_items=1000, weight=weight)
        trainer_fmf.run(train=train, test=test, iters=iters, batch_size=batch_size)
        evaluator = Evaluator(test=test, rare=False, name='fmf')
        evaluator.evaluate()
        evaluator_rare = Evaluator(test=test_rare, rare=True, name='fmf')  # evaluate the performance on rare items
        evaluator_rare.evaluate()

    if args.model == 'RMF':
        latent_dim = 200
        regu_lam = 0.00001
        iters = 300
        batch_size = 2**15
        learning_rate = 0.005
        warnings.filterwarnings("ignore")
        train, test, test_rare, num_users, num_items = yahoo_load()
        trainer = RmfTrainer(latent_dim=latent_dim, regu_lam=regu_lam, iters=iters, batch_size=batch_size,
                             learning_rate=learning_rate, num_users=15400, num_items=1000)
        trainer.run(train=train, test=test, iters=iters, batch_size=batch_size)
        evaluator = Evaluator(test=test, rare=False, name='rmf')
        evaluator.evaluate()
        evaluator_rare = Evaluator(test=test_rare, rare=True, name='rmf')  # evaluate the performance on rare items
        evaluator_rare.evaluate()

    if args.model == 'FMF-MPE':
        latent_dim = 200
        regu_lam = 0.00001
        iters = 300
        batch_size = 2 ** 15
        learning_rate = 0.005
        weight = 1
        warnings.filterwarnings("ignore")
        train, test, test_rare, num_users, num_items = mpe_yahoo_load()
        trainer_fmf = FmfTrainer(latent_dim=latent_dim, regu_lam=regu_lam, iters=iters, batch_size=batch_size,
                                 learning_rate=learning_rate, num_users=15400, num_items=1000, weight=weight, mpe=True)
        trainer_fmf.run(train=train, test=test, iters=iters, batch_size=batch_size)
        evaluator = Evaluator(test=test, rare=False, name='fmf-mpe')
        evaluator.evaluate()
        evaluator_rare = Evaluator(test=test_rare, rare=True, name='fmf-mpe')  # evaluate the performance on rare items
        evaluator_rare.evaluate()

    if args.model == 'RMF-MPE':
        latent_dim = 200
        regu_lam = 0.00001
        iters = 300
        batch_size = 2**15
        learning_rate = 0.005
        warnings.filterwarnings("ignore")
        train, test, test_rare, num_users, num_items = mpe_yahoo_load()
        trainer = RmfTrainer(latent_dim=latent_dim, regu_lam=regu_lam, iters=iters, batch_size=batch_size,
                             learning_rate=learning_rate, num_users=15400, num_items=1000, mpe=True)
        trainer.run(train=train, test=test, iters=iters, batch_size=batch_size)
        evaluator = Evaluator(test=test, rare=False, name='rmf-mpe')
        evaluator.evaluate()
        evaluator_rare = Evaluator(test=test_rare, rare=True, name='rmf-mpe')  # evaluate the performance on rare items
        evaluator_rare.evaluate()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Luo")
    parser.add_argument('--m', dest='model', default='RMF-MPE')  # choose from (BPR, FMF, RMF, FMF-MPE, RMF-MPE)
    args = parser.parse_args()
    main(args)

