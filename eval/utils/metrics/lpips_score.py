import lpips
import cv2

def compute_lpips_score(img_paths_1, img_paths_2):
    ## Initializing the model
    loss_fn = lpips.LPIPS(net='alex',version="0.1")
    loss_fn.cuda()
    
    i = 0
    dists = []
    for i in range(len(img_paths_1)):
        img0 = lpips.load_image(img_paths_1[i])
        img0 = cv2.resize(img0, (64, 64))
        img1 = lpips.load_image(img_paths_2[i])
        img1 = cv2.resize(img1, (64, 64))
        img0 = lpips.im2tensor(img0) # RGB image from [-1,1]
        img1 = lpips.im2tensor(img1)

        img0 = img0.cuda()
        img1 = img1.cuda()

        # Compute distance
        dist01 = loss_fn.forward(img0,img1)
        # print('%s: %.3f'%(file,dist01))
        dists.append(dist01.item())
        i += 1
    
    average = sum(dists) / len(dists)
    return average
    