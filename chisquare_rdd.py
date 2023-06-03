from pyspark import SparkContext
import re
import json

# Create a SparkContext
sc = SparkContext(appName="ChiSquareCalculation")

# Load the input file from HDFS
input_file = sc.textFile("hdfs:///user/dic23_shared/amazon-reviews/full/reviews_devset.json")

# Here we read the list of stopwords that must be ignored from the review text stored as global variable
stop_words = open('stopwords.txt').readlines()
stop_words = [w.replace('\n', '') for w in open('stopwords.txt').readlines() ]

# Function to preprocess the review text and extract terms
def preprocess_review(line):
    
    data = json.loads(line)
    review_text = data["reviewText"]
    category = data["category"]
    
    # Simplify word tokens, unigram preprocessing
    review_words_list = re.split('[^a-zA-Z<>^|]+', review_text)  # splitting words by spaces, characters etc.
    review_words_list = [f.lower() for f in review_words_list] # lower case letters to make items uniform
    review_words_list = [f for f in review_words_list if len(f) > 1 ] # exclude terms with only one letter
    review_words_list = [str(w) for w in review_words_list if w not in stop_words ] # Filter stop words
    review_words_list = list(set(review_words_list)) # Remove duplicated words
    
    return [(category,word) for word in review_words_list]

#Preprocess the review text and extract terms
terms_rdd = input_file.flatMap(preprocess_review)

'''Output after the flatMap
('Patio_Lawn_and_Garde', 'insight')
('Patio_Lawn_and_Garde', 'things')
('Patio_Lawn_and_Garde', 'open')
'''

#Calculate term frequency per category
#term_freq_rdd = terms_rdd.map(lambda x: (x,1)).reduceByKey(lambda x, y:x +y)
term_freq_rdd = terms_rdd.map(lambda x: ((x[0], x[1]), 1)).reduceByKey(lambda x, y: x + y)
'''Output after reducer
(('Patio_Lawn_and_Garde', 'insight'), 1)
(('Patio_Lawn_and_Garde', 'things'), 31)
(('Patio_Lawn_and_Garde', 'open'), 14)
(('Patio_Lawn_and_Garde', 'raichlen'), 1)
'''

#Calculate term frequency per term across all categories
#term_freq_all_rdd = term_freq_rdd.map(lambda x: (x[0][1], x[1])).reduceByKey(lambda x, y: x + y)
term_freq_all_rdd = term_freq_rdd.map(lambda x: (x[0][0], [(x[0][1], x[1])])) \
                                .reduceByKey(lambda x, y: x + y) \
                                .mapValues(lambda x: {term: freq for term, freq in x})

'''
Output after the reducer reducer
"Patio_Lawn_and_Garde"	{"interpret":1,"yum":1,"provided":8,"simple":17,"gift":21,"easy":142,
'''

# Join the term frequency per category and term frequency per RDDs
term_freq_mapped_rdd = term_freq_all_rdd.flatMap(lambda x: [(term, [x[0], freq]) for term, freq in x[1].items()])

term_freq_grouped_rdd = term_freq_mapped_rdd.groupByKey()
term_freq_grouped_mapped_rdd = term_freq_grouped_rdd.map(lambda x: (x[0], list(x[1])))

'''Output after mapping
abandonment	[['Book', 20], ['Digital_Music', 1], ['CDs_and_Vinyl', 1], ['Kindle_Store', 3], ['Movies_and_TV', 2]]
abandons	[['Book', 3], ['Movies_and_TV', 1]]
abaord	[['Movies_and_TV', 1]]
'''

""" Loading the precomputed dictionary of total number of reviews per category """
def load_cat_dict():
    cat = open( 'op.txt', 'r').readlines()
    cats = [eval(l.split('\t')[0].replace("'",'')) for l in cat]
    num = [int(l.split('\t')[1].replace('\n','')) for l in cat]
    cat_dic = dict(zip(cats,num))
    cat_dic['N'] = sum(cat_dic.values())
    
    return cat_dic 

# storing as global variable
cat_dict = load_cat_dict()
N = sum(cat_dict.values())

# abandonment	[['Book', 20], ['Digital_Music', 1], ['CDs_and_Vinyl', 1], ['Kindle_Store', 3], ['Movies_and_TV', 2]]
# abandons	[['Book', 3], ['Movies_and_TV', 1]]
# abaord	[['Movies_and_TV', 1]]


def test(data):
    chi_dict = {}
    term, freqs = data
    term_freq_all = sum(freq[1] for freq in freqs)

    for cat_freq in freqs:
        category = cat_freq[0]
        A = cat_freq[1]
        B = term_freq_all - A
        #B = sum(freq[1] for freq in cat_freq) - A
        C = cat_dict[category] - A
        D = N - A - B - C
        if (A + C) * (B + D) * (A + B) * (C + D)==0:
            chi = 0.0
        else:
            chi = (N * ((A * D) - (B * C)) ** 2) / ((A + C) * (B + D) * (A + B) * (C + D))
        chi_dict[category] = (term, chi)
        
    return chi_dict

# Calculate chi-square values for each term in each category
chi_square_dict_rdd = term_freq_grouped_mapped_rdd.map(test)

# Sort the terms within each category by their chi-square values in descending order
sorted_chi_square_rdd = chi_square_dict_rdd.flatMap(lambda x: x.items()) \
                                          .sortBy(lambda x: (x[0], -x[1][1]), ascending=True)

# Group the sorted terms by category
grouped_sorted_chi_square_rdd = sorted_chi_square_rdd.groupByKey()

# Take the top 75 elements from each line of the sorted RDD
#top_75 = grouped_sorted_chi_square_rdd.take(75)
top_75 = grouped_sorted_chi_square_rdd.mapValues(lambda x: list(x)[:75])

for category_terms in top_75.collect():
    category = category_terms[0]
    terms = category_terms[1]
    print(f"{category}: ", end="")
    for term in terms:
        print(term, end=" ")
    print()  # Print a newline after each category


# Stop the SparkContext
sc.stop()