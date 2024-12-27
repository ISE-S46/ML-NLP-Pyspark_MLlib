from graphviz import Digraph
from IPython.display import Image, display

def visualize_decision_tree():
    dot = Digraph(comment='Decision Tree for Loan Approval Rules')

    # Root decision: Credit Score >= 700
    dot.node('A', 'credit_score >= 700?')
    dot.node('B', 'annual_income > 50,000?')
    dot.node('C', 'Class: Approved')
    dot.edge('A', 'B', label='yes')
    dot.edge('B', 'C', label='yes')

    # Branch for Credit Score >= 650 and Debt-to-Income Ratio < 35
    dot.node('D', 'credit_score >= 650?')
    dot.node('E', 'debt_to_income_ratio < 35?')
    dot.edge('A', 'D', label='no')
    dot.edge('D', 'E', label='yes')
    dot.edge('E', 'C', label='yes')  # Approved

    # Branch for Credit Score < 600 or Debt-to-Income Ratio > 45
    dot.node('F', 'credit_score < 600 or debt_to_income_ratio > 45?')
    dot.node('G', 'Class: Rejected')
    dot.edge('D', 'F', label='no')
    dot.edge('F', 'G', label='yes')

    # Default case: Pending
    dot.node('H', 'Class: Pending')
    dot.edge('F', 'H', label='no')

    # Render and display the diagram
    dot.render('/content/loan_decision_tree', format='png', view=False)
    display(Image(filename='/content/loan_decision_tree.png'))

visualize_decision_tree()