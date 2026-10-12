import torch

def calculate_brightness(img) -> float:
    """
    Calculate the average brightness of a grayscale image using PyTorch.

    Args:
        img: A 2D list where each element represents a pixel value between 0-255.

    Returns:
        The average brightness rounded to two decimal places, or -1 for invalid inputs.
    """
    # Write your code here
    if not img :
        return -1
    chicun = all(len(i) == len(img[0]) for i in img)
    if chicun == False:
        return -1
    img_t = torch.tensor(img,dtype=torch.float)
    if img_t.max()>255 or img_t.min()<0:
        return -1
    else:
        return (torch.mean(img_t).round(decimals=2)).item()

