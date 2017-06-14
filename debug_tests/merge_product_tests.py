import tensorflow  as tf
import numpy as np
from keras import backend as K
from keras.layers import merge

def merge_product_old(generator_output, orig_img_input):
    #return ValueError("Merge protuct is not fully implemented")
    #generator_output = K.permute_dimensions(generator_output, (3,0,1,2))
    #orig_img_input = K.permute_dimensions(orig_img_input, (3,0,1,2))

    #H = []
    #for i in range(orig_img_input.shape[0]):
    #    for j in range(generator_output.shape[0]):
    #        new_prod = merge([orig_img_input[i], generator_output[j]], mode='mul')
    #        H.append(new_prod)

    #H_merged = tf.stack(H, axis=1)
    #H_merged = K.permute_dimensions(H_merged, (0,2,3,1))
    #print(orig_img_input.shape)
    #orig_img_input = K.transpose(orig_img_input)
    #print(orig_img_input.shape)
    aux_orig_img_input = orig_img_input
    aux_generator_output = generator_output


    for i in range(aux_generator_output.shape[3]-1):
        orig_img_input =  merge([orig_img_input, aux_orig_img_input], mode='concat',concat_axis=1)
    
    for i in range(aux_orig_img_input.shape[3]-1):
        generator_output = merge([generator_output, aux_generator_output], mode='concat', concat_axis=3)
    
    
    print(orig_img_input.shape)
    #generator_output = K.transpose(generator_output)
    #generator_output = K.permute_dimensions(generator_output, (3,1,2,0))
    #print(generator_output.shape)
    
    #print(H_merged.shape)
    return orig_img_input, generator_output


def merge_product(generator_output, orig_img_input):
    aux_orig_img_input = orig_img_input
    aux_generator_output = generator_output

    for i in range(aux_generator_output.shape[3]-1):
        orig_img_input =  merge([orig_img_input, aux_orig_img_input], mode='concat',concat_axis=3)
    
    for i in range(aux_orig_img_input.shape[3]-1):
        generator_output = merge([generator_output, aux_generator_output], mode='concat', concat_axis=1)
    
    generator_output = K.permute_dimensions(generator_output, (0,2,3,1))
    print(orig_img_input.shape)
    print(generator_output.shape)

    return orig_img_input, generator_output

if __name__ == "__main__":
    n_clases = 5
    max_val = 255
    generator_output = K.placeholder(shape=(5, 5, n_clases))
    orig_img_input = K.placeholder(shape=(5, 5, 3))

    npc = np.random.randint(max_val, size=(1,2,2,3))/255.
    tfc = tf.Variable(npc) # Use variable 

    npc2 = np.random.randint(20, size=(1,2,2,2))/1.0
    tfc2 = tf.Variable(npc2) # Use variable 
    

    res1, res2 = merge_product(tfc2, tfc)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print tfc
        print(tfc.eval())
        print ""
        print(tfc2.eval())
        print ""
        print(res1.eval())
        print ""
        print(res2.eval())