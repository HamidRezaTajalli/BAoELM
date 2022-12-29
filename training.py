from elm_versions import elm, pca_transformed, pca_initialization, pruned_elm, drop_elm


def trainer(elm_type: str, dataset: str, trigger: str, hdlyr_size: int):
    elm_dict = {'poelm': elm.ELMClassifier, 'elm-pca': pca_transformed.PCTClassifier,
                'pca-elm': pca_initialization.PCIClassifier, 'pruned-elm': pruned_elm.PrunedClassifier,
                'drop-elm': drop_elm.DropClassifier}

    if elm_type.lower() == 'poelm':
        elm = elm.ELMClassifier(hidden_layer_size=hdlyr_size, activation='sigm')
        elm.fit(x_train, y_train)
        out = elm.predict(X)




