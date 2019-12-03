import random
import pandas as pd

def pre_process_training_data(training_data):
    df = training_data
    
    # lowercase all characters in the training_data and remove trailing/leading spaces
    df = df.applymap(lambda s:s.strip().lower() if type(s) == str else s)

    # a dict which counts the number of times a product name occurs in the product_name column in the training data
    product_name_value_counts = df['product_name'].value_counts()
    product_name_value_counts = dict([(k,{'count':v,'type' : 'product_name'}) for (k,v) in product_name_value_counts.items()])

    # a dict which counts the number of times a producer occurs in the producer column in the training data
    producer_value_counts = df['producer'].value_counts()
    producer_value_counts = dict([(k,{'count':v,'type' : 'producer'}) for (k,v) in producer_value_counts.items()])

    # a dict which counts the number of times a quantity occurs in the quantity column in the training data
    quantity_value_counts = df['quantity'].value_counts()
    quantity_value_counts = dict([(k,{'count':v,'type' : 'quantity'}) for (k,v) in quantity_value_counts.items()])

    return product_name_value_counts, producer_value_counts, quantity_value_counts

def extract_brand_names(product_descriptions):
    """
    Example input:
    ["Birger jarl röd 28% Falbygdens ost kg",
     "Duschcreme Lenande Barnängen 250 ml",
     "Färskostfyllda Lökar Färskvaruhuset Lösvikt",
     "Grillade grönsaker Eget kök 1kg"]

     Example output:
    ["Falbygdens ost",
     "Barnängen",
     "Färskvaruhuset",
     "Eget kök"]
    """

    # pre-process training data
    product_name_value_counts, producer_value_counts, quantity_value_counts = pre_process_training_data(pd.read_csv('training_data.csv', header=0))
    # collect all values into one list
    training_data_values = [product_name_value_counts, producer_value_counts, quantity_value_counts]
    
    brands = [] # final list of brands of all the product desc in the test set.
    brands_df = pd.DataFrame(columns=['A'])

    # iterate thru every description in the test set.
    for description in product_descriptions:
        # remove trailing and leading spaces and lower case all characters.
        description = description.rstrip().lower()
        # split a description into words
        desc_split = description.split(" ")
        
        # number of words
        end = len(desc_split)

        # Basic idea is to segment the description into three slices (of any length) using two pivots, 
		# 1. pivotA that starts from the beginning of the description and moves L-> R; 
        # 2. PivotB that starts from the end of the description and moves R -> L
        pivotA = 0
        pivotB = end
        
        # Max segmentation score achieved so far for this description
        max_segmentation_score = 0
        # predicted brand of this description 
        brand = ""
        
        # move PivotA from 1 to end
        for i in range(end+1):
            pivotA = i
            pivotB = end - 1

            # for each PivotA, decrement PivotB, check if there is 3-segment and find the score of the segment.
            while pivotB >= pivotA:
                # score of the current segmentation
                segmentation_score = 0
                # Brand can belong to any segment.
                # Eg. Tin Box Oreo 350 g
                # segment_1 = Tin Box, segment_2 = Oreo, segment_3 = 350 g
                # In certain descriptions, it is possble that the brand can belong to either segment.
                # Hence, we need to find out in which segment, the brand name is more likely to belong to. 
                brand_is_in_which_segment =  -1
                
                # Condition to check , a 3-segment can be formed using the current vales of PivotA and PivotB
                if pivotB >= pivotA and pivotA > 0 and end - 1 - pivotB > 0:
                    
                    # get the 3 segments
                    segment_1 = desc_split[0:pivotA]
                    segment_2 = desc_split[pivotA:pivotB+1]
                    segment_3 = desc_split[pivotB+1:end+1]
                    
                    # join the words in each segment, to be able to search in the pre-processed training data
                    segment_1 = ' '.join(segment_1)
                    segment_2 = ' '.join(segment_2)
                    segment_3 = ' '.join(segment_3)

                    # collect all segments into one list
                    segments = [segment_1, segment_2, segment_3]
                    
                    # Now, we check each segment in the training data, i.e. in product name, producer, quantity and get its 
                    # corresponding count of occurence. 
                    # max_value will hold the maximum occurence count of segment_1 as a product_name / producer / quantity 
                    max_value = -1
                    # we define three entities : product name, producer, quantity
                    # the value in "which_entity" points to any of the item in "training_data_values" 
                    which_entity = "" 
                    
                    if segment_1 in product_name_value_counts and product_name_value_counts[segment_1]['count'] > max_value:
                        max_value = product_name_value_counts[segment_1]['count']
                        which_entity =  0 # if which_entity is 0, it means segment_1 is more often found in product name.
                    
                    if segment_1 in producer_value_counts and producer_value_counts[segment_1]['count'] > max_value:
                        max_value = producer_value_counts[segment_1]['count']
                        which_entity =  1 # if which_entity is 1, it means segment_1 is more often found in producer.
                    
                    if segment_1 in quantity_value_counts and quantity_value_counts[segment_1]['count'] > max_value:
                        max_value = quantity_value_counts[segment_1]['count']
                        which_entity =  2 # if which_entity is 2, it means segment_1 is more often found in quantity.
                    
                    if max_value != -1:
                        if which_entity == 1:
                            brand_is_in_which_segment = 0 # this index points to the "segments" list
                        
                        # add the occurence count of segment_1 to the segmentation score
                        # the occurance count is taken from "training_data_values" list using the value in which_entity
                        segmentation_score += training_data_values[which_entity][segment_1]['count']
                        
                    # Do the above process for segment_2
                    max_value = -1
                    which_entity = ""
                    
                    if segment_2 in product_name_value_counts and product_name_value_counts[segment_2]['count'] > max_value:
                        max_value = product_name_value_counts[segment_2]['count']
                        which_entity =  0
                    
                    if segment_2 in producer_value_counts and producer_value_counts[segment_2]['count'] > max_value:
                        max_value = producer_value_counts[segment_2]['count']
                        which_entity =  1
                    
                    if segment_2 in quantity_value_counts and quantity_value_counts[segment_2]['count'] > max_value:
                        max_value = quantity_value_counts[segment_2]['count']
                        which_entity =  2
                    
                    if max_value != -1:
                        if which_entity == 1:
                            brand_is_in_which_segment = 1
                        # add the occurence count of segment_2 to the segmentation score
                        segmentation_score += training_data_values[which_entity][segment_2]['count']
                    
                    # Do the above process for segment_3
                    max_value = -1
                    which_entity = ""
                    
                    if segment_3 in product_name_value_counts and product_name_value_counts[segment_3]['count'] > max_value:
                        max_value = product_name_value_counts[segment_3]['count']
                        which_entity =  0
                    
                    if segment_3 in producer_value_counts and producer_value_counts[segment_3]['count'] > max_value:
                        max_value = producer_value_counts[segment_3]['count']
                        which_entity =  1
                    
                    if segment_3 in quantity_value_counts and quantity_value_counts[segment_3]['count'] > max_value:
                        max_value = quantity_value_counts[segment_3]['count']
                        which_entity =  2
                    
                    if max_value != -1:
                        if which_entity == 1:
                            brand_is_in_which_segment = 2
                        # add the occurence count of segment_3 to the segmentation score
                        segmentation_score += training_data_values[which_entity][segment_3]['count']
                    
                    # all segments have been checked.
                    # now compare score of current segmentation with the max segmentation score found so far.
                    if segmentation_score >= max_segmentation_score:
                        max_segmentation_score = segmentation_score

                        # if the brand has not been found in the training dataset, then brand_is_in_which_segment is likely to be -1
                        # In this case, select the value in segment_2 as the BRAND. Not sure if this is the best alternative!
                        brand = segments[brand_is_in_which_segment] if brand_is_in_which_segment != -1 else segment_2
                                
                # decrement PivotB
                pivotB = pivotB - 1

        # append this description's brand to the final list of brands.
        brands.append(brand)
        brands_df = brands_df.append({'A': brand}, ignore_index=True)

    brands_df.to_csv('predictions.csv', index=False)
    return brands

    # For now selects a random word as brand name.
    # return [random.choice(description.split(" ")) for description in product_descriptions]
