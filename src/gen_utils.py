

import torch
import torch.nn as nn
import json
import pickle
import matplotlib.pyplot as plt




# Load the weights from the JSON file
def load_weights(filename="weights/LoRA_weights_pet.json"):
    with open(filename, 'r') as file:
        weights = json.load(file)
    return weights





def custom_print(*args, sep=' ', end='\n', filename=f"output_file.txt"):
    print(*args, sep=sep, end=end)
    try:
        with open(filename, 'a') as file:  # Open file in append mode
            string = ' '.join(str(item) for item in args) + '\n'  # Join all items with a newline
            file.write(string)  # Write the concatenated string to the file
    except Exception as e:
        print(f"Error while appending to file: {e}")
        
        
        
        
        
def save_dict_to_file(dictionary, filename="LoRA_weights_baseline.json"):
    try:
        with open(filename, 'w') as file:
            json.dump(dictionary, file, indent=4)
        print(f"Dictionary successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the dictionary: {e}")



def save_dict_to_file_binary(dictionary, filename):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Dictionary successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the dictionary: {e}")


    
    
def check_fidelity(model, data_loader, stored_output_file = "weights/victim_model_outputs.json", device='cuda:0'):
    # Ensure model is in evaluation mode
    model.eval()  
    
    # Load saved outputs
    with open(stored_output_file, "r") as f:
        model_1_outputs = json.load(f)

    # Initialize fidelity counters
    matching_predictions = 0
    total_predictions = 0

    # Compare predictions
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)         # ("cuda" if torch.cuda.is_available() else "cpu")
            labels = labels.to(device)         # ("cuda" if torch.cuda.is_available() else "cpu")
            preds_2 = model(inputs).logits.argmax(dim=1).cpu().tolist()

            # Get corresponding predictions from model_1
            preds_1 = model_1_outputs[i]["predictions"]

            # Calculate matching predictions
            for p1, p2 in zip(preds_1, preds_2):
                if p1 == p2:
                    matching_predictions += 1
                total_predictions += 1

    # Calculate and display fidelity
    fidelity = (matching_predictions / total_predictions) * 100
    print(f"Fidelity between model_1 and model_2: {fidelity:.2f}%", end='\t************\t')
    print(f"total prediction = {total_predictions} and matched = {matching_predictions}")
    
    return fidelity

       


               
def plot_fidelity_graph(X_Axis, Y_Axis):  
    # Compute the maximum accuracy for each weight
    fidelities = [max(y_values) for y_values in Y_Axis]

    
    # Plot the graph
    plt.figure(figsize=(8, 6))
    plt.plot(X_Axis, fidelities, marker='o', linestyle='-', color='b', label='sub_model_fidelity')
    
    # # for baseline
    # base_acc = [91.94, 91.94, 91.94, 91.94, 91.94, 91.94]
    # plt.plot(weights_, base_acc, marker='*', label='victim_acc', color='red')

    # Add labels, title, and legend
    plt.xlabel('Removed Weights')
    plt.ylabel('Fidelity(%)')
    plt.title('Removed_Weights vs. Fidelity')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()