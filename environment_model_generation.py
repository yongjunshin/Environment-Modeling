import argparse
import glob
import os
import datetime

from src.BehaviorCloning1TickTrainer import BehaviorCloning1TickTrainer
from src.BehaviorCloningEpisodeTrainer import BehaviorCloningEpisodeTrainer
from src.LineTracerEnvironmentModelGRU import LineTracerEnvironmentModelGRU
from src.LineTracerVer1 import LineTracerVer1
from src.DatasetBuilder import *
import time


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--source_file", type=str, nargs='+',
                    help="<Mandatory (or -d)> field test data source files (list seperated by spaces)", default=["data/ver1_fixed_interval/ver1_ft_60_30.csv", "data/ver1_fixed_interval/ver1_ft_60_40.csv"])
parser.add_argument("-d", "--source_directory", type=str,
                    help="<Mandatory (or -f)> filed test data directory (It will read only *.csv files.)", default=None)
possible_mode = ["bc_1tick_noDagger", "bc_1tick_Dagger", "bc_episode"]
parser.add_argument("-m", "--mode", type=str,
                    help="model generation algorithm among "+str(possible_mode)+" (default: bc_1tick_noDagger)", default='bc_episode')
parser.add_argument("-l", "--history_length", type=int,
                    help="history length (default: 100)", default=100)
parser.add_argument("-e", "--epochs", type=int,
                    help="num epochs (default: 30)", default=30)
parser.add_argument("-b", "--batch_size", type=int,
                    help="mini batch size (default: 128)", default=1000)
parser.add_argument("-md", "--max_dagger", type=int,
                    help="maximum number of dagger", default=10)
parser.add_argument("-dt", "--dagger_threshold", type=float,
                    help="dagger operation flag threshold", default=0.02)
parser.add_argument("-dm", "--distance_metric", type=str,
                    help="history distance metric ['ed', 'wed', 'md', 'wmd', 'dtw'] (default: dtw)", default='dtw')
parser.add_argument("-el", "--episode_length", type=int,
                    help="episode length (default: same with history length)", default=None)
parser.add_argument("-ms", "--manual_seed", type=int,
                    help="manual seed (default: random seed)", default=None)
parser.add_argument("-er", "--experiment_repeat", type=int,
                    help="experiment repeat (default: 1)", default=1)
parser.add_argument("-els", "--episode_loss", type=str,
                    help="episode loss function ['mse', 'mdtw'](default: mse)", default='mdtw')

args = parser.parse_args()


# Running mode selection and wrong input handling
def mode_selection(args):
    if args.source_file is None and args.source_directory is None:
        print("[Error] Source files (-f) or a source directory (-d) must be given. Refer help (-h).")
        quit()
    if args.source_file is not None and args.source_directory is not None:
        print("[Error] Only source files (-f) or a source directory (-d) must be given at once. Refer help (-h).")
        quit()
    if args.mode not in possible_mode:
        print("[Error] Wrong mode was given. Select among " + str(possible_mode) + ". Refer help (-h).")
        quit()

    if args.mode == "bc_1tick_noDagger":
        mode = 0
    elif args.mode == "bc_1tick_Dagger":
        mode = 1
    elif args.mode == "bc_episode":
        mode = 2

    return mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mode = mode_selection(args)
source_files = args.source_file
source_directory = args.source_directory
history_length = args.history_length
epochs = args.epochs
batch_size = args.batch_size
if mode == 1:
    dagger_on = True
else:
    dagger_on = False
max_dagger = args.max_dagger
dagger_threshold = args.dagger_threshold
distance_metric = args.distance_metric
if mode == 2:
    if args.episode_length is None:
        episode_length = history_length
    else:
        episode_length = args.episode_length
else:
    episode_length = args.episode_length

random_seed = args.manual_seed
if random_seed == None:
    random_seed = int(time.time())

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
set_seed(random_seed)


experiment_repeat = args.experiment_repeat
episode_loss = args.episode_loss

# Input specification
print("Environment model generation (start at", datetime.datetime.now(), ")")
print("=====Input specification=====")
print("Available device:", device)
print("Source data:", source_files)
print("History length:", history_length)
print("Epochs:", epochs)
if mode == 0:
    print("Environment model generation algorithm: 1-tick Behavior Cloning without DAgger")
elif mode == 1:
    print("Environment model generation algorithm: 1-tick Behavior Cloning with DAgger")
    print("Maximum number of DAgger execution:", max_dagger)
    print("DAgger execution thresthold (prediction threshold):", dagger_threshold)
    print("History distance metric:", distance_metric)
elif mode == 2:
    print("Environment model generation algorithm: Episode Behavior Cloning")
    print("Episode length:", episode_length)
    print("Episode loss:", episode_loss)


print("Random seed:", random_seed)

print("Experiment repeat:", experiment_repeat)
print("=====(end)=====")
print()


for e in range(experiment_repeat):
    print("====={0}th environment model generation process=====".format(e))
    print("Step 1: Source data reading")
    # Source data reading
    if source_directory is not None:
        source_files = glob.glob(os.path.join(source_directory, "*.csv"))

    raw_dfs = []
    for file in source_files:
        raw_dfs.append(pd.read_csv(file, index_col='time')[:300])
    print("--data size:", [raw_df.shape for raw_df in raw_dfs])

    print("Step 2: Data normalization")
    noramlized_nparrays, scaler = normalize_dataframes_to_nparrays(raw_dfs)

    print("Step 3: Build train/test/validation dataset")
    # Build train/test/validation dataset
    train_dataloaders, test_dataloaders, validation_dataloaders = build_train_test_validation_dataset(noramlized_nparrays, mode, history_length, episode_length, batch_size, device)

    print("--train dataset shape:", [str(loader.dataset.x.shape) +'->' + str(loader.dataset.y.shape) for loader in train_dataloaders])
    print("--test dataset shape:", [str(loader.dataset.x.shape) +'->' + str(loader.dataset.y.shape) for loader in test_dataloaders])
    print("--validation dataset shape:", [str(loader.dataset.x.shape) +'->' + str(loader.dataset.y.shape) for loader in validation_dataloaders])

    print("Step 4: Build environment model")
    # instantiate a moddel
    input_dim = 2
    hidden_dim = 16
    num_layers = 2
    output_dim = 1
    model = LineTracerEnvironmentModelGRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, device=device)
    model.to(device=device)
    print("--environment model summary:", model)

    print("Step 5: Build system under test")
    # instantiate a moddel
    line_tracer = LineTracerVer1(scaler)

    print("Step 5: Train environment model")
    # train the environment model
    if mode == 0 or mode == 1:
        trainer = BehaviorCloning1TickTrainer(device=device, sut=line_tracer)
        training_loss, dagger_count = trainer.train(model=model, epochs=epochs, train_dataloaders=train_dataloaders,
                                                    test_dataloaders=test_dataloaders, dagger=dagger_on,
                                                    max_dagger=max_dagger, dagger_threshold=dagger_threshold,
                                                    dagger_batch_size=batch_size, distance_metric=distance_metric)
        print("--training loss:", training_loss)
        print("--dagger count:", dagger_count)
    elif mode == 2:
        trainer = BehaviorCloningEpisodeTrainer(device=device, sut=line_tracer)
        training_loss = trainer.train(model=model, epochs=epochs, train_dataloaders=train_dataloaders,
                                                    test_dataloaders=test_dataloaders, loss_metric=episode_loss)
        print("--training loss:", training_loss)




    print("=====(end)=====")