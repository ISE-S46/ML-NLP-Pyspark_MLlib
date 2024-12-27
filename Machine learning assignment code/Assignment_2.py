# Practicum 2. Visualization Diagram of Rule-Based Classifier
# Install Graphviz system package
# !apt-get -y install graphviz

# Install Python package for Graphviz
# !pip install graphviz

from graphviz import Digraph
from IPython.display import Image, display

def predict_class(data):
    if data["age"] > 30 and data["income"] > 50000:
        return "buy"
    elif data["marital_status"] == "single" and data["income"] < 30000:
        return "don't buy"
    elif data["age"] < 25 and data["income"] > 40000:
        return "buy"
    else:
        return "unknown"

def visualize_decision_tree():
    dot = Digraph(comment='Decision Tree from Rules')

    dot.node('A', 'age > 30?')
    dot.node('B', 'income > 50,000?')
    dot.node('C', 'Class: Buy')
    dot.edge('A', 'B', label='yes')
    dot.edge('B', 'C', label='yes')

    dot.node('D', 'marital_status = "single"?')
    dot.node('E', 'income < 30,000?')
    dot.node('F', 'Class: Don\'t buy')
    dot.edge('D', 'E', label='yes')
    dot.edge('E', 'F', label='yes')

    dot.node('G', 'age < 25?')
    dot.node('H', 'income > 40,000?')
    dot.node('I', 'Class: Buy')
    dot.edge('G', 'H', label='yes')
    dot.edge('H', 'I', label='yes')

    dot.node('J', 'Class: Unknown')
    dot.edge('A', 'J', label='no')
    dot.edge('D', 'J', label='no')
    dot.edge('G', 'J', label='no')

    dot.render('/content/decision_tree', format='png', view=False)

    display(Image(filename='/content/decision_tree.png'))

visualize_decision_tree()