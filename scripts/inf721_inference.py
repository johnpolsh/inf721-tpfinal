from inf721_model import *
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Loading model for tests
bottleneckLayerDetail = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
]
test_model = OurObjectDetectionNet(bottleneckLayerDetail)
optimizer = torch.optim.Adam(test_model.parameters(), lr=0.01)
load_model_for_resume("last-run.pth", test_model, optimizer)

def make_confusion_matrix(model):
    true_labels = []
    predicted_labels = []
    model.eval()

    with torch.no_grad():
        for inps, lbls in test_dataloader:
            vinputs = inps.to(device)
            vlabels = lbls.to(device)
            if torch.cuda.is_available():
                voutputs = nn.parallel.data_parallel(model, vinputs)
            else:
                voutputs = model(vinputs)
            _, predicted = torch.max(voutputs, 1)
            true_labels.extend(vlabels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    return confusion_matrix(true_labels, predicted_labels)

cm = make_confusion_matrix(test_model)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(classes)))
disp.plot()
plt.show()
