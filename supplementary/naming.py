imgs = {
    "Simple Form": (
        "simple_form.png",
        "This form will automatically determine the best parameters for the pipeline. New users are recommended to start with this form."
        ),
    "Advanced Form": (
        "advanced_form.png",
        "This form can be used to specify the parameters of the model. The parameters are described in the documentation."
        ),
     "Data Validation Page": (
        "data_validation.png",
        "This is a tool to assist the user in determining whether their data is valid for use with this service. The user can upload a CSV file and the service will return a list of errors, if any exist."
        ),
    "Request ID Page": (
        "request_id.png",
        "After a request is submitted, the user will be presented with this page. The request ID can be saved and used to view the results after the pipeline executes."
        ),
    "Results Landing Page": (
        "results_landing_page.png",
        "When the service has completed running the pipeline, the user will be presented with this page. The user can choose to view customizable visualizations or download the data for their own use."
        ),
    "Boxplot": (
        "ex_boxplot.png",
        "This plot shows the prediction error for each set of k models from the cross validation process. Low values indicate better model fit, and narrower boxes indicate less variance in the model fit."
        ),
    "Lineplot": (
        "ex_lineplot.png",
        "This plot provide an alternative view to the Boxplot figure. The x-axis shows which of the k models the data point corresponds to. The y-axis shows the prediction error for that model. The color of the line indicates which set of k models the data point corresponds to."
        ),
    "Best Model": (
        "ex_best_model.png",
        "This plot shows the prediction error for the single best (according to the specified 'ranking metric') model from each set of the k models constructed during k-fold cross validation. The model has been used to predict on a held-out set of the data."
        ),
    "2D PCA": (
        "ex_2D_PCA.png",
        "This plot allows the user to determine how well the model performs over different regions of the data. Red points indicate that the model has predicts a point poorly, while blue points indicate the model is accurate for that point. The 2 most important dimensions from Principal Component Analysis (PCA) have been extracted for use in this graph."
        ),
    "2D TSNE": (
        "ex_2D_TSNE.png",
        "This plot allows the user to determine how well the model performs over different regions of the data. Red points indicate that the model has predicts a point poorly, while blue points indicate the model is accurate for that point. The 2 most important dimensions from t-Distributed Stochastic Neighbor Embedding (t-SNE) have been extracted for use in this graph."
        ),
    "3D PCA #1": (
        "ex_3D_PCA_1.png",
        "This plot allows the user to determine how well the model performs over different regions of the data. Red points indicate that the model has predicts a point poorly, while blue points indicate the model is accurate for that point. The 2 most important dimensions from Principal Component Analysis (PCA) have been extracted for use in this graph."
        ),
    "3D PCA #2": (
        "ex_3D_PCA_2.png",
        "This is an alternate view of the 3D PCA plot."
        ),
    "3D TSNE": (
        "ex_3D_TSNE.png",
        "This plot allows the user to determine how well the model performs over different regions of the data. Red points indicate that the model has predicts a point poorly, while blue points indicate the model is accurate for that point. The 2 most important dimensions from t-Distributed Stochastic Neighbor Embedding (t-SNE) have been extracted for use in this graph."
        ),
    "Quantity Curve": (
        "ex_quant_curve.png",
        "This plot provides the user information about the optimal amount of training data to provide their models. The blue line shows training error, or how well the model predicts its own training data. The green line shows test error, or how well the model predicts data it has not been trained on. The two line will eventually converge, which indicates that the model will not benefit from any more training data."
        ),
    "Region PCA": (
        "ex_region_PCA.png",
        "This plot allows the user to determine how well the model performs over different discretized regions of the data. Red points indicate that the model has predicts a point poorly, while blue points indicate the model is accurate for that point. The 2 most important dimensions from Principal Component Analysis (PCA) have been extracted for use in this graph."
        ),
    "Region TSNE": (
        "ex_region_TSNE.png",
        "This plot allows the user to determine how well the model performs over different discretized regions of the data. Red points indicate that the model has predicts a point poorly, while blue points indicate the model is accurate for that point. The 2 most important dimensions from t-Distributed Stochastic Neighbor Embedding (t-SNE) have been extracted for use in this graph."
        ),
}

with open("f.out", "w") as f:
    for k, (path, description) in imgs.items():
        print(f"![{k}](./images/{path})\n\n<br>\n{description}\n<br>\n{'-'*90}\n<br><br><br><br><br><br>\n\n\n", file=f)