import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import clim


def show_plots(acc, val_acc, loss, val_loss):
    """
    Utility function to plot training and validation losses and accuracies.

    Keyword arguments:
    :acc list: training accuracy readings
    :val_acc list: validation accuracy readings
    :loss list: training loss readings
    :val list: validation loss readings
    """
    epochs = range(1, len(acc)+1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    max_acc = max(val_acc)
    plt.axhline(y=max_acc, color='r', linestyle='-')
    plt.title(f'Training and validation accuracy\n Max val acc: {max_acc:.2f}')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """
    Plot the decision function for a 2D SVC.

    :param model: trained support vector classifier object.
    :param ax: matplotlib axis to use for plotting.
    :param plot_support: whether to plot the support vectors.
    """
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
                levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                    model.support_vectors_[:, 1],
                    s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



def plot_class_regions_for_classifier_subplot(clf, X, y, X_test, y_test,
                                              title, subplot,
											  target_names = None,
                                              plot_decision_regions = True):
    """
    Plot decision regions within a subplot for a trained classifier
    in the space of two of the features.

    :param clf: trained classifier
    :param X: 2-D array of the two chosen features
    :param y
    """

    numClasses = np.amax(y) + 1
    color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF']
    color_list_bold = ['#EEEE00', '#000000', '#00CC00', '#0000CC']
    cmap_light = ListedColormap(color_list_light[0:numClasses])
    cmap_bold  = ListedColormap(color_list_bold[0:numClasses])

    h = 0.03
    k = 0.5
    x_plot_adjust = 0.1
    y_plot_adjust = 0.1
    plot_symbol_size = 50

    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    # x coordinates and y coordinates at each of grid points to plot
    x2, y2 = np.meshgrid(np.arange(x_min-k, x_max+k, h),
                         np.arange(y_min-k, y_max+k, h))

    P = clf.predict(np.c_[x2.ravel(), y2.ravel()])
    P = P.reshape(x2.shape)

    if plot_decision_regions:
        subplot.contourf(x2, y2, P, cmap=cmap_light, alpha = 0.8)
    # plot features
    subplot.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s=plot_symbol_size,
                    edgecolor = 'black')
    subplot.set_xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
    subplot.set_ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
    # plot test observations
    if (X_test is not None):
        subplot.scatter(X_test[:, 0], X_test[:, 1],
                        c=y_test, cmap=cmap_bold,
                        s=plot_symbol_size, marker='^', edgecolor = 'black')
        train_score = clf.score(X, y)
        test_score  = clf.score(X_test, y_test)
        title = title + "\nTrain score = {:.2f}, Test score = {:.2f}".format(train_score,
                                                                             test_score)

    subplot.set_title(title)

    # create legend for plot
    if (target_names is not None):
        legend_handles = []
        for i in range(0, len(target_names)):
            patch = mpatches.Patch(color=color_list_bold[i], label=target_names[i])
            legend_handles.append(patch)
        subplot.legend(loc=0, handles=legend_handles)


def visualize_classifier(model, X, y, ax=None, cmap='rainbow', title=None):
    """
    Visualize the classification boundaries of a classifier in two features.

    :param model: sklearn-like model object
    :param X: feature array
    :param y: target array
    :param ax: optional axes object on which to plot
    :param cmap: optional colour map for plot
    :param title: optional title for plot
    """
    ax = ax or plt.gca()

    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    #ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap,
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
    if title is not None:
        ax.set_title(title)
