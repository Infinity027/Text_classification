import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import random as rd
import argparse
import os

class CharacterSegmentation:
    def __init__(self, min_line_height=10, min_area=100, draw_plot=False):
        self.spaces = []
        self.chars = []
        self.min_height = min_line_height
        self.draw_plot = draw_plot
        self.min_area = min_area #minimum size of character present in words
    
    def normalize(self, arr,mode='minmax'):
        if mode=="minmax":
            return (arr - np.min(arr))/(np.max(arr) - np.min(arr))
        else:
            return (arr - np.mean(arr))/np.std(arr)
        
    def divideaspeak(self, img, dir=1, th = 0, extra_space=True):
        """
        img: input binary image
        dir: sum axis
        """
        pixel_sum = np.sum(img, axis=dir)

        #conver the range into O-1
        pixel_sum = self.normalize(pixel_sum,mode='minmax')

        #depend on threshold find the peak point
        peaks = np.where(pixel_sum>th)[0]
        #split on peak point
        locs = np.split(peaks, np.where(np.diff(peaks) > 1)[0] + 1)
        final_locs = []
        for loc in locs:
            if dir==0 and len(loc)>2:
                final_locs.append([loc[0],loc[-1]])
            
            #if lines height is too small discarded that lines
            elif len(loc)>self.min_height:
                if extra_space==True:
                    width = round(np.sqrt(loc[-1] - loc[0]))
                    final_locs.append([loc[0]-width,loc[-1]+width])
                else:
                    final_locs.append([loc[0],loc[-1]])
        return final_locs
    
    def segment_characters(self, image):
        """
        image: binary image
        min_area: minimum a rea of a character can be
        """
        # _, binary_img = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
        # # Apply morphological operations to improve segmentation
        # kernel = np.ones((3, 3), np.uint8)
        # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours of characters
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours from left to right
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        character_images = []
        # Loop through detected contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out small regions (noise)
            if w * h > self.min_area:
                char_img = image[:, x:x+w]  # Extract character
                character_images.append(char_img)

        return character_images
    
    def get_img(self, image_path, c=2):
        gray_img = cv2.imread(image_path,0)
        th, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        gray_img = 1-gray_img
        print("Thresold detect:",th)
        lines_loc = self.divideaspeak(binary_img)

        if len(lines_loc)==0:
            print("No lines found or Image is too small -- try different images")
            exit(1)
        # print(lines_loc)
        
        for i, l_loc in enumerate(lines_loc):
            line_height = l_loc[-1]-l_loc[0]
            #increase characters width
            # print("line height:",line_height)
            b_img = cv2.dilate(binary_img[l_loc[0]:l_loc[-1]], 
                               np.ones((3,3), np.int8), 
                               iterations=c)
            
            #find words in lines and remember spaces
            words_loc = self.divideaspeak(b_img, dir=0, extra_space=False)
            h = len(self.chars)
            for w_loc in words_loc:
                charaters = self.segment_characters(binary_img[l_loc[0]:l_loc[-1],w_loc[0]:w_loc[-1]])
                self.chars.extend(charaters)
                self.spaces.append(len(self.chars))
            if self.draw_plot:
                self.draw_line_word_ch(binary_img[l_loc[0]:l_loc[-1]],
                                       words_loc,
                                       self.chars[h:],
                                       k=i)
        for idx in range(len(self.chars)):
            diff = self.chars[idx].shape[0]-self.chars[idx].shape[1]
            if diff>0:
                self.chars[idx] = cv2.copyMakeBorder(self.chars[idx], 0, 0, diff//2, diff//2, cv2.BORDER_CONSTANT,value=[0,0,0])
            elif diff<0:
                self.chars[idx] = cv2.copyMakeBorder(self.chars[idx], abs(diff//2), abs(diff//2), 0, 0, cv2.BORDER_CONSTANT,value=[0,0,0])
        return self.chars, self.spaces
    def draw_line_word_ch(self,line,word_locs,chars,k=1):
        words = []
        for loc in word_locs:
            words.append(line[:,loc[0]:loc[-1]])
        save_name = 'result/line_'+str(k)+'.png'
        self.plot(line,words,chars,save_name)
    
    def plot(self, line, words, chars, save_name="plot.png"):
        fig = plt.figure(figsize=(len(words),4))

        # Define a GridSpec for layout control
        gs = gridspec.GridSpec(3, 1, figure=fig)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax1.set_title("Line Segmentation")
        ax2.set_title("Word Segmentaion")
        ax3.set_title("Character Segmentaion")
        ax1.imshow(line,cmap="gray")
        ax1.axis("off")
        ax2.axis("off")
        ax3.axis("off")

        gs_word = gridspec.GridSpecFromSubplotSpec(1, len(words), subplot_spec=gs[1])
        for i, word in enumerate(words):
            ax_w=fig.add_subplot(gs_word[i])
            ax_w.imshow(word,cmap="gray")
            ax_w.axis("off")
        gs_ch = gridspec.GridSpecFromSubplotSpec(1, len(chars), subplot_spec=gs[2])
        for i, ch in enumerate(chars):
            ax_w=fig.add_subplot(gs_ch[i])
            ax_w.imshow(ch,cmap="gray")
            ax_w.axis("off")
        plt.tight_layout
        plt.savefig(save_name)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default='data', help="Text image path")
    parser.add_argument("--c", type=int, default=4, help="dilate mode iteration")
    parser.add_argument("--min_area", type=int, default=100, help="minimum area to find out a character")
    parser.add_argument("--draw_plot", type=bool, default=False, help="Draw plot of Images of characters")
    args = parser.parse_args()
    # Check if resume path exists
    if not os.path.exists(args.image_path):
        print(f"Warning: Image Path '{args.model_path}' does not exist.")
        exit(1)  
    return args

def character_plot(images, save_name="plot.png"):
    num_of_img = len(images)
    row = int(np.ceil(np.sqrt(num_of_img)))
    if row*(row-1)<=num_of_img:
        col = row-1
    else:
        col = row
    plt.figure(figsize=(row,col))
    for i,image in enumerate(images):
        plt.subplot(row,col,i+1)
        plt.imshow(image,cmap="gray")
        plt.axis("off")
    plt.savefig(save_name)
    plt.close()
    
if __name__=="__main__":
    args = parse_args()
    cs = CharacterSegmentation(min_line_height=10, min_area=args.min_area, draw_plot=args.draw_plot)
    chars, spaces = cs.get_img(args.image_path,c=args.c)
    random_chars = rd.choices(chars,k=20)
    print("Total Characters:",len(chars))
    print("Spaces position:",*spaces)
    character_plot(random_chars,"result/random_char.png")



