# Description: This module implements all the EDA steps settled upon
# based on looser work in eda.ipynb. It also generates the EDA figures
# and puts them in the figures directory.
import data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split

# Make global variables for the feature names, expert labeled
# images, and filepath to figs directory.
COORDS = ["y", "x"]
RAD_ANGLES = ["DF", "CF", "BF", "AF", "AN"]
SYN_FTRS = ["NDAI", "SD", "CORR"]
EXP_LAB = "Expert Label"
FTRS = COORDS + SYN_FTRS + ["Radiance Angle " + ang for ang in RAD_ANGLES] \
    + [EXP_LAB]
MISR_EXP_IM_NUM = [12791, 13257, 13490]
FIGS = "../figs/"

def make_pixel_df():
    """
    Load in the data using the make_data function without
    removing labels. Don't save patches since it is just based 
    on the images and not relevant to cleaning and EDA. Make
    Pandas dataframe of images and return the result.

    Parameters: (None)

    Returns:
        pd.DataFrame: A Pandas dataframe of the pixels and their
            features.
    """
    images_long, _ = data.make_data(remove_labels=False)

    # Set the length of images_long (164) to a constant.
    im_count = len(images_long)

    # Initialize a dictionary to store concatenated numpy arrays of
    # each feature across all images.
    pixel_dict = {}

    for idx, ftr in enumerate(FTRS):
        # Concatenate all the numpy arrays tied to the feature
        # and store the result. For the unlabeled images, impute NaN's
        # in the expert label feature.
        pixel_dict[ftr] = np.concatenate([images_long[im][:,idx]
                                        if idx < images_long[im].shape[1] 
                                        else np.array([np.nan] * 
                                                      len(images_long[im]))
                                        for im in range(im_count)])
        
    # Add key-value pair to pixel_dict to store the original image index
    # of each.
    pixel_dict["Image"] = np.concatenate([np.array([im] * len(images_long[im]))
    for im in range(im_count)])
    
    # Return Pandas dataframe of dictionary.
    return pd.DataFrame(pixel_dict)


def make_heatmap(pixel_df):
    """
    Make a correlation heatmap of the main features and
    writes it to "heatmap.png" in the figs directory.

    Parameters:
        pixel_df (pd.DataFrame): A Pandas dataframe of the pixels
        and their features.
    
    Returns:
        None
    """
    # Exclude expert labels and image indexes when making heatmap.
    pixel_df_unlabeled = pixel_df.drop(columns=[EXP_LAB, "Image"])

    # Abbreviate radiance angle names to avoid figure getting cut off
    # when saved.
    pixel_df_unlabeled.rename(lambda ftr: "Rad. Angle " + ftr[-3:]
                          if "Radiance Angle" in ftr else ftr, axis="columns",
                          inplace=True)
    
    # Make a correlation heatmap (we don't need to set numeric_only in
    # corr to True since every entry in pixel_df_unlabeled is numeric anyway),
    # rounding the displayed correlations to 2 decimal places.
    sns.heatmap(pixel_df_unlabeled.corr(), cmap="coolwarm", fmt=".2f", 
                annot=True)

    # Add a title to the plot.
    plt.title("Correlation Heatmap Between Main Features (All Images)")
    # Change size of x and y ticks since they are too big by default.
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Use tight_layout since figure keeps getting cut off when saved.
    plt.tight_layout()

    # Save the figure in the figures directory.
    plt.savefig(FIGS + "heatmap.jpg")
    plt.close()


def plot_exp_lab(pixel_df):
    """
    Plot expert labels on (X, Y)-axis for all three labeled images
    and save the resulting figure in the figs directory.

    Parameters:
        labeled_df (pd.DataFrame): A Pandas dataframe of the pixels
        and their features for only the labeled images.
    
    Returns:
        None
    """
    # List labeled image indexes.
    labeled_idx = labeled_df["Image"].unique()

    # Make a dictionary mapping the labeled indexes to the true MISR
    # image numbers.
    labeled_im_name_map = dict([(k, v) for k, v in zip(labeled_idx, 
                                                       MISR_EXP_IM_NUM)])

    # Split the labeled data frame into each of the three labeled images.
    labeled_images = [labeled_df[labeled_df["Image"] == labeled_idx[im]]
                    for im in range(len(labeled_idx))]
    
        # Make a colormap for the expert labels matching Yu 2008.
    cmap = mcolors.ListedColormap(["gray", "black", "white"])

    # Make a 1 x 3 grid of subplots.
    fig, axs = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)

    # For each image, plot the expert labels for the presence or
    # absence of clouds in each images according to a map 
    # (i.e. use the X, Y coordinates).
    for im in range(len(labeled_images)):
        axs[im].scatter(x=labeled_images[im][COORDS[0]], 
                        y=labeled_images[im][COORDS[1]], 
                            c=labeled_images[im]["Expert Label"],
                            cmap=cmap)
        
        # Set name of each subplot to be MISR number of image.
        axs[im].set_title(f"Image {labeled_im_name_map[labeled_idx[im]]}")
        
    # Make shared axis labels as the x- and y-coordinates and
    # an overall title.
    fig.supxlabel("x-coordinate")
    fig.supylabel("y-coordinate")
    fig.suptitle("Expert Labels by Pixel in 3 Labeled Images")
        
    # Add legend handles to make color legend for entire plot.
    legend_handles = [
        mpatches.Patch(color="white", label="Cloud"),
        mpatches.Patch(color="gray", label="Clear"),
        mpatches.Patch(color="black", label="Unlabeled"),
    ]

    # Use legend_handles to make legend for full figure.
    fig.legend(handles=legend_handles, loc="outside right upper")

    # Use tight layout to avoid overlap, using 15 percent of
    # the space on the right for the legend, and show the plot.
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save figure.
    plt.savefig(FIGS + "expert_labels.jpg")
    plt.close()


def plot_rad_dist(rad_df):
    """
    Make boxplots of radiance angle distributions for all 5 angles
    and save the figure.

    Parameters:
        rad_df (pd.DataFrame): A Pandas dataframe of the pixels
        and their radiance features for only the labeled images.
    
    Returns:
        None        
    """
    # Make a boxplot of the radiance features.
    sns.boxplot(data=rad_df, fliersize=0, 
                color="cyan")
    plt.xlabel("Radiance Angle")
    plt.ylabel("Radiance")
    plt.title("Radiance Distribution by Angle in Labeled Images")

    # Save figure.
    plt.savefig(FIGS + "radiance_dist.jpg")
    plt.close()


def plot_rad_dist_by_label(rad_df_tot_lab):
    """""
    Make boxplots of Radiance Angle distributions for all 5 angles
    split between whether the pixels were labeled cloudy or clear
    and save the figure.

    Parameters:
        rad_df_tot_lab (pd.DataFrame): A Pandas dataframe of the pixels
        and their radiance features for only the labeled images, keeping
        the expert labels.
    
    Returns:
        None        
    """
    # Make a boxplot of the radiance features separated by
    # cloud label status, using a long format version of the 
    # dataframe as required by the boxplot function.
    sns.boxplot(data=rad_df_tot_lab.melt(id_vars=["Expert Label"], 
                                        value_vars=RAD_ANGLES, 
                                        var_name="Radiance Angle",
                                        value_name="Radiance"),
                                        x="Radiance Angle",
                                        y="Radiance",
                                        fliersize=0, 
                                        hue="Expert Label")

    # Add x and y axes and title.
    plt.xlabel("Radiance Angle")
    plt.ylabel("Radiance")
    plt.title("Radiance Distribution by Angle and Label in Labeled Images")

    # Save figure.
    plt.savefig(FIGS + "radiance_dist_by_label.jpg")
    plt.close()


def make_syn_ftr_table(syn_df_cloud, syn_df_clear):
    """
    Make and save a table of the difference in means (cloud - clear)
    among the synthetic features (NDAI, SD, CORR) between pixels labeled
    cloudy and clear, and also include the p-value of t-tests of
    independence between the groups for each synthetic feature.

    Parameters:
        syn_df_cloud (pd.DataFrame): A Pandas dataframe of synthetic
            features and expert labels for the cloudy labeled pixels.
        syn_df_clear (pd.DataFrame): A Pandas dataframe of synthetic
            features and expert labels for the clear labeled pixels.

    Returns:
        None
    """
    # Initialize a dictionary to store tuples of mean differences and p-values
    # for t-tests of independence when comparing CORR, SD, and NDAI 
    # between cloud and clear expert labeled pixels.
    diff_and_p_val_dict = {}

    # Perform t-tests to compare CORR, SD, and NDAI between cloud and clear
    # labeled pixels. Also calculate mean differences (cloud - clear)
    # and add them both to dictionary.
    # Round mean difference to 5 decimal places and convert p-values to
    # scientific notation rounded to 5 decimal places.
    for syn_ftr in SYN_FTRS:
        # Calculate mean difference and p-value.
        mean_diff = "{:.5f}".format(syn_df_cloud[syn_ftr].mean()
                        - syn_df_clear[syn_ftr].mean())
        p_val = ttest_ind(syn_df_cloud[syn_ftr], syn_df_clear[syn_ftr]).pvalue
        
        # If p-value is just 0, impute it as 0 rather than 0.00000e+0,
        # which is redundant.
        if p_val != 0:
            p_val = '{:0.5e}'.format(p_val)
        
        # Add both mean difference and p value to dictionary.
        diff_and_p_val_dict[syn_ftr] = (mean_diff, p_val)

        # Convert dictionary to pandas dataframe, transposing to make
    # synthetic features be the rows.
    diff_and_pval_df = pd.DataFrame(diff_and_p_val_dict).T

    # Change column names.
    diff_and_pval_df.columns = ["Difference in Means", "T-test P-value"]

    # Turn off axes and border.
    fig, axs = plt.subplots(1, 1)
    axs.axis("tight")
    axs.axis("off")

    # Choose row and column colors.
    color = "lavender"

    # Make table.
    plt.table(diff_and_pval_df, loc="center",
            colColours=[color] * diff_and_pval_df.shape[1],
            rowColours=[color] * diff_and_pval_df.shape[0],
            )
    
    # Save figure.
    plt.savefig(FIGS + "syn_ftr_dist_by_label.jpg")
    plt.close()


def train_val_test_split(tot_labeled_df, random_state, save=False):
    """
    Split data into training, validation, and testing sets with
    60, 20, and 20 percent of the labeled image pixels in each.
    To mimick the real-world prediction setting but also capture
    as much variation as possible in the labeled images, we randomly
    assign pixels from each image such that they match the 60/20/20 
    training/validation/testing split. This is a medium between
    group-based and random splits. If asked to save, write the result to 
    code/train_val_test_split.npz as three numpy arrays for 
    training, validation, and testing. Return a tuple of the six
    dataframes (i.e., X_train, y_train, X_val, y_val, X_test, y_test).

    Parameters:
        tot_labeled_df (pd.DataFrame): A Pandas dataframe of the pixels
            and their features for only the labeled images, excluding
            the unlabeled pixels.
        random_state (int): An integer to define the random
            states of our train/test splits for reproducibility.
        save (bool): Whether to write the result to a .npz file.
    
    Returns:
        Tuple[pd.DataFrame]: A tuple of feature/target pairs
            of Pandas dataframes associated with training, validation,
            and testng in that order.
    """
    # Initialize a list of lists to store the image-based splits.
    splits_by_im = []

    # Iterate through expert labeled images and do the same to each.
    for im in tot_labeled_df["Image"].unique():
        # Randomly split labeled_df into 80/20 train/test split.
        tt_split = train_test_split(tot_labeled_df[tot_labeled_df["Image"] 
                                                   == im], 
                                    train_size=0.8,
                                    random_state=random_state)

        # Split training set into new training set and validation set with
        # 80/20 split.
        tv_split = train_test_split(tt_split[0], train_size=0.75,
                                    random_state=random_state)

        # Combine results to get 60/20/20 train/val/test split.
        splits_by_im.append([tv_split[0], tv_split[1], tt_split[1]])
    
    # Combine results across the expert labeled images.
    all_splits = [pd.concat([splits_by_im[im][split]
                         for im in range(len(splits_by_im))])
                         for split in range(len(splits_by_im[0]))]

    # Drop image and coordinate labels from all three splits to 
    # avoid training upon them, as long as they are in the splits
    # in the first place.
    drop_list = ["Image"] + COORDS
    drop_list = list(filter(lambda x: x in all_splits[0].columns, drop_list))
    all_splits = [df.drop(columns=drop_list) 
                  for df in all_splits]
    
    # If asked to save, write the train/val/test splits to a 
    # .npz file.
    if save:
        # Extract and name the train/val/test splits as numpy arrays.
        train = all_splits[0].to_numpy(dtype=np.float32)
        val = all_splits[1].to_numpy(dtype=np.float32)
        test = all_splits[2].to_numpy(dtype=np.float32)

        np.savez_compressed("train_val_test_split.npz", train=train,
                            validation=val,
                            test=test)
    
    # Make a list to store the X and y splits.
    xy_splits = []

    # Iterate through the splits, split them into X and y (i.e., features
    # and label), and append the results to xy_splits.
    for im in range(len(tot_labeled_df["Image"].unique())):
        xy_splits.append(all_splits[im].iloc[:, :-1])
        xy_splits.append(all_splits[im].iloc[:, -1:])

    # Return xy_splits as a tuple.
    return tuple(xy_splits)


# Run the functions needed to generate the figures.
if __name__ == "__main__":
    # Get rid of SettingWithCopyWarning I deal with later.
    pd.options.mode.chained_assignment = None
    
    # Make pixel_df and heatmap.
    pixel_df = make_pixel_df()
    make_heatmap(pixel_df)

    # Filter the data frame just to the labeled images.
    labeled_df = pixel_df[pixel_df[EXP_LAB].notna()]

    # Make expert labels plot.
    plot_exp_lab(labeled_df)

    # Make list of radiance features.
    rad_ftrs = ["Radiance Angle " + ang for ang in RAD_ANGLES]

    # Filter the labeled dataframe just for the
    # radiance features.
    rad_df = labeled_df[rad_ftrs]

    # Change the column names to get rid of the redundant
    # "Radiance Angle" at the start.
    rad_df.columns = rad_df.columns.str.removeprefix("Radiance Angle ")

    # Make radiance distribution plot.
    plot_rad_dist(rad_df)

    # Next, drop the unlabeled pixels from the labeled dataframe.
    tot_labeled_df = labeled_df[labeled_df[EXP_LAB] != 0]

    # Generate train/val/test splits for the labeledimages
    # and write them to code/train_val_test_split.npz.
    # Use a random state of 1 for reproducibility.
    train_val_test_split(labeled_df, 1, save=True)

    # Recode the labels as Cloud and Clear.
    tot_labeled_df[EXP_LAB] = tot_labeled_df[EXP_LAB].replace({
        1: 'Cloud',
        -1: 'Clear'
    })

    # Regenerate radiance dataframe with only labeled pixels
    # but keep labels this time.
    rad_df_tot_lab = tot_labeled_df[rad_ftrs + [EXP_LAB]]

    # Change the column names to get rid of the redundant
    # "Radiance Angle" at the start.
    rad_df_tot_lab.columns = \
        rad_df_tot_lab.columns.str.removeprefix("Radiance Angle ")
    
    # Make plot of radiance distribution by label.
    plot_rad_dist_by_label(rad_df_tot_lab)

    # Generate synthetic feature dataframe with only labeled pixels
    # and keep labels.
    syn_df_tot_lab = tot_labeled_df[SYN_FTRS + [EXP_LAB]]

    # Split the dataframe based on expert label.
    syn_df_cloud = syn_df_tot_lab[syn_df_tot_lab[EXP_LAB] == "Cloud"]
    syn_df_clear = syn_df_tot_lab[syn_df_tot_lab[EXP_LAB] == "Clear"]

    # Make the synthetic feature distribution comparison table.
    make_syn_ftr_table(syn_df_cloud, syn_df_clear)

