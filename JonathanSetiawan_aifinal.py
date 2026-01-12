# Name: Jonathan Setiawan
# Student ID: 26002404663

# declare all imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import scikit learn
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
# import dataset used to convert pandas dataframes into a format compatible with huggingface's setfit
from datasets import Dataset
# import setfit used for fine-tuning the model using contrastive learning
from setfit import SetFitModel, SetFitTrainer
# import cosine similarity loss for contrastive learning (how close embeddings are in a vector space in terms of angle)
from sentence_transformers.losses import CosineSimilarityLoss
import torch

# file path configurations and constants
DATA_PATH = "data/processed/augmented_dataset.csv" # path to the processed dataset
MODEL_ID = "allenai/scibert_scivocab_uncased" # scibert model identifier
OUTPUT_DIR = "models/final_submission_model" # path to output the final model
FIGURES_DIR = "results/figures" # path to save figures
K_FOLDS = 5 # number of folds for cross-validation
SEED = 42 # random seed

# check if dir exists, if not create it
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_and_preprocess_data(file_path):
    # function to load and preprocess the data, input is the file path to the dataset we just defined

    # load dataset
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path) # read the csv file into a pandas dataframe
    
    # drop rows with missing text or labels
    df = df.dropna(subset=['text', 'label'])
    
    # encode labels into the scientific fields for classification: bioinformatics, neuroscience, materials science
    le = LabelEncoder()
    df['label_idx'] = le.fit_transform(df['label']) # create a new column with encoded labels
    
    # print data info
    print(f"Data loaded successfully. Total samples: {len(df)}")
    print(f"Classes found: {list(le.classes_)}")
    
    return df, le

def train_and_evaluate_fold(fold_idx, train_df, val_df, device):
    # function to train and evaluate a single fold, input is the fold index, 
    # training dataframe, validation dataframe, and device (it's basically cpu or gpu)

    # note that dataframes are like tables with rows and columns used to store data in pandas

    # just print header
    print(f"\n--- Training Fold {fold_idx+1}/{K_FOLDS} ---")
    
    # convert to huggingface format from pandas dataframe for both the train and val sets
    train_ds = Dataset.from_pandas(train_df[['text', 'label_idx']].rename(columns={'label_idx': 'label'}))
    val_ds = Dataset.from_pandas(val_df[['text', 'label_idx']].rename(columns={'label_idx': 'label'}))
    
    # load the scibert model with a setfit header
    model = SetFitModel.from_pretrained(MODEL_ID)
    model.to(device)
    
    # initialize the setfit trainer to fine tune the model
    trainer = SetFitTrainer(
        # these are the parameters for constrastive learning
        model=model,
        train_dataset=train_ds, # ds stands for dataset
        eval_dataset=val_ds, 
        loss_class=CosineSimilarityLoss, # use cosine similarity loss for contrastive learning
        metric="accuracy", # use accuracy as the evaluation metric
        batch_size=16, # batch size is the number of samples processed before the model is updated, we make it 16 because of memory constraints
        num_iterations=3, # number of pairs to generate for contrastive learning
        num_epochs=1, # number of epochs to train the model
        column_mapping={"text": "text", "label": "label"} # mapping the columns
    )
    
    # train the model
    trainer.train()
    
    # evaluate the model
    metrics = trainer.evaluate()
    
    # get the predictions for metrics report
    preds = model.predict(val_df['text'].tolist())
    
    # return the model, metrics, predictions, and true labels
    return model, metrics, preds, val_df['label_idx'].values

def plot_confusion_matrix(y_true, y_pred, target_names, save_path):
    # function to plot confusion matrix, input is true labels, predicted labels, 
    # target names (which are the class names), and save path

    # define confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    
    # plot the confusion matrix
    plt.figure(figsize=(10, 8))
    # we make a heatmap for the visualization graph, cm stands for confusion matrix, annot=True is to show annotations, 
    # fmt='d' stands for format as integers, cmap='Blues' is the color map used, xticklabels and yticklabels are the class names
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    # add the labels for x and y axis and title
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Final Model Confusion Matrix')
    # save the figure in the path specified before
    plt.savefig(save_path)
    # print the confirmation message
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def plot_class_performance(report_dict, save_path):
    # function to plot the classification performance, input is the report dictionary and save path

    # we declare the classes and metrics we want to plot, in this case it's precision, recall, and f1-score for each class
    classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']] # filter out non-class keys
    metrics = ['precision', 'recall', 'f1-score']
    
    data = [] # create an empty list to store the data for plotting
    for cls in classes: # loop through each class
        for metric in metrics: # loop through each metric
            data.append({
                'Class': cls, # class name
                'Metric': metric.capitalize(), # metric name
                'Score': report_dict[cls][metric] # score value
            })
            
    df_plot = pd.DataFrame(data) # convert the data into a pandas dataframe
    
    plt.figure(figsize=(12, 6)) # set the diagram size to 12x6 inches
    sns.barplot(x='Class', y='Score', hue='Metric', data=df_plot, palette='viridis') # create a bar plot using seaborn
    # the x axis is the class, y axis is the score, hue means different colros for different metrics, for data we're using the dataframe plot we created
    # the pallete is the color scheme used and we use viridis because it looks nice
    plt.title('Model Performance by Class') # name the title
    plt.ylim(0, 1.1) # set the range of the y axis
    plt.grid(axis='y', linestyle='--', alpha=0.7) # add gridlines
    plt.legend(loc='lower right') # set where the legend is located
    plt.savefig(save_path) # save the figure at the path we defined
    print(f"Class performance plot saved to {save_path}") # print this message to confirm it worked
    plt.close() # done

def plot_tsne_clusters(model_path, df, le, save_path, n_samples=1000):
    # function to plot t-SNE clusters of the embeddings, input is the model path, 
    # dataframe, label encoder, save path, and number of samples, in this case number of samples corelate to how many points we want to plot

    print("Generating t-SNE cluster visualization...") # print message

    # load the model
    model = SetFitModel.from_pretrained(model_path)
    
    # if the dataframe length is larger than n samples, we take a random sample (from a total set of 42 states) of n samples to make it faster
    if len(df) > n_samples:
        df_sample = df.sample(n=n_samples, random_state=42)
    else:
        df_sample = df
        
    texts = df_sample['text'].tolist() # get the texts from the dataframe by converting it to a list
    labels = df_sample['label_idx'].values # get the labels from the dataframe as a numpy array
    class_names = le.inverse_transform(labels) # convert the encoded labels back to original class names
    
    # get the embeddings from the model after encoding the texts
    embeddings = model.encode(texts)
    
    # t-SNE is something we do to reduce the dimensions of the embeddings to 2D for visualization, 
    # so from a high dimensional space to a 2D space
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto') # set parameters for t-SNE as 
    # n_components=2 refer to the number of dimensions we want to reduce to which means we want 2D output, random_state=42 
    # means that there are 42 different states we can have, perplexity=30 is a balance between local and global aspects of the data and it
    # refers to the number of nearest neighbors considered when positioning points in the reduced space, init='pca' means we initialize the t-SNE with PCA which is 
    # another dimensionality reduction technique and stands for principal component analysis, learning_rate='auto' means the learning rate is set automatically

    X_2d = tsne.fit_transform(embeddings) # fit and transform the embeddings to 2D space using t-SNE
    
    # plot the t-SNE results in a graph
    plt.figure(figsize=(12, 8)) # set the graph size to 12x8 inches
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=class_names, palette='deep', s=60, edgecolor='k') # create a scatter plot using seaborn
    # s=60 means the size of the points is 60, edgecolor='k' means the edge color of the points is black
    plt.title('t-SNE Visualization of Scientific Abstract Embeddings') # set the title of the plot
    plt.legend(title='Domain') # set the legend title
    plt.savefig(save_path) # save the figure at the path we defined
    print(f"t-SNE plot saved to {save_path}") # print this message to confirm it worked
    plt.close() # plot done

def main():
    # main function to run the whole thing
    print("====================================================")
    print("   AI2025 FINAL ASSIGNMENT: SCIENTIFIC CLASSIFIER   ")
    print("====================================================\n")
    
    # Check if there's a gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution Device: {device}")
    
    # load and prepare the data, call function
    df, le = load_and_preprocess_data(DATA_PATH)
    
    # setup the k-fold cross-validation using sklearn's kfold
    # the parameters are number of splits, whether we want to shuffle the data before splitting which is used to reduce bias,
    # and random state=seed which means we want to have reproducible results
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    
    # we make lists to store the accuracies for each fold, predictions, and true labels
    fold_accuracies = []
    all_preds = []
    all_trues = []
    
    # loop for cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
        # for each fold index in the kfold split of the dataframe, we get the train and validation indices
        # then we use those indices to get the actual dataframes for training and validation

        train_df = df.iloc[train_idx] # get the training dataframe using the train indices
        val_df = df.iloc[val_idx] # get the validation dataframe using the validation indices
        
        model, metrics, preds, trues = train_and_evaluate_fold(fold_idx, train_df, val_df, device) 
        # call the function to train and evaluate the fold so we can get the model, metrics, predictions, and true labels
        
        fold_accuracies.append(metrics['accuracy']) # append the accuracy for this fold to the list
        all_preds.extend(preds) # extend the all predictions list with the predictions from this fold
        all_trues.extend(trues) # extend the all true labels list with the true labels from this fold
        
        # Save the best model (using the first fold as a baseline for this assignment)
        if fold_idx == 0:
            print(f"Saving final model to {OUTPUT_DIR}...")
            model.save_pretrained(OUTPUT_DIR)

    # loop will repeat for all folds based on the number of k folds we defined earlier, so in this case 5 folds, 
    # then after all folds are done we just print some messages for confirmation
            
    # performance report, print result messages
    print("\n" + "="*40) # make a separator line using = 40 times
    print("         FINAL PERFORMANCE REPORT         ")
    print("="*40)
    
    print(f"\nK-Fold Cross-Validation Results (k={K_FOLDS}):") # print the k fold cross validation results
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})") # print the mean accuracy and standard deviation of the accuracies across folds
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_trues, all_preds, target_names=le.classes_)) # print the detailed classification report
    
    # generate report dictionary used for visualization
    report_dict = classification_report(all_trues, all_preds, target_names=le.classes_, output_dict=True)
    
    print("\nGenerating Visualizations...") # just print this message so we know where we're at
    
    # plot the confusion matrix
    plot_confusion_matrix(all_trues, all_preds, le.classes_, 
                         os.path.join(FIGURES_DIR, "final_confusion_matrix.png"))
                         
    # plot the class performance
    plot_class_performance(report_dict, 
                          os.path.join(FIGURES_DIR, "final_class_performance.png"))
                          
    # plot the t-SNE clusters using the best model saved
    plot_tsne_clusters(OUTPUT_DIR, df, le, 
                      os.path.join(FIGURES_DIR, "final_tsne_clusters.png"))
    
    print("Models and visualizations have been saved successfully.") # print messages to confirm

if __name__ == "__main__":
    main()
