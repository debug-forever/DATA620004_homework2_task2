import os.path as osp
import mmcv
import cv2
import matplotlib.pyplot as plt
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.structures import DetDataSample
import numpy as np

_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

def draw_detections(img, detections, score_thr, class_names):
    for bbox, score, label in zip(detections.bboxes.cpu().numpy(),
                                  detections.scores.cpu().numpy(),
                                  detections.labels.cpu().numpy()):
        if score > score_thr:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label_text = class_names[label] if label < len(class_names) else f'Unknown({label})'
            text = f'{label}: {score:.2f}'
            cv2.putText(img, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img

def draw_segmentations(img, masks, class_names):
    colors = [(0, 255, 0)] * len(class_names)
    for mask, color in zip(masks, colors):
        img[mask > 0] = img[mask > 0] * 0.5 + np.array(color) * 0.5
    return img

def draw_segmentations_with_labels(img, masks, labels, scores, class_names):
    """
    Draw instance segmentation masks with labels and scores on the image.
    Args:
        img: Input image (numpy array).
        masks: Segmentation masks (numpy array of shape [N, H, W]).
        labels: List of label IDs corresponding to masks.
        scores: List of scores corresponding to masks.
        class_names: List of class names corresponding to label IDs.
    Returns:
        img: Image with segmentation masks, labels, and scores drawn.
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for mask, label, score in zip(masks, labels, scores):
        color = colors[label % len(colors)]  # Cycle through colors
        img[mask > 0] = img[mask > 0] * 0.5 + np.array(color) * 0.5  # Blend mask with image
        # Find the center of the mask to place the label and score
        y, x = np.where(mask > 0)
        if len(x) > 0 and len(y) > 0:
            center_x, center_y = int(np.mean(x)), int(np.mean(y))
            label_text = class_names[label] if label < len(class_names) else f'Unknown({label})'
            text = f'{label}: {score:.2f}'
            cv2.putText(img, text, (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

def visualize_proposals_and_results(model_maskrcnn, model_sparsercnn, img_path, score_thr):
    """
    Visualize proposals, detection results, and instance segmentation results for both models.
    Args:
        model_maskrcnn: Mask R-CNN model.
        model_sparsercnn: Sparse R-CNN model.
        img_path: Path to the input image.
        score_thr: Score threshold for detections.
    """
    # Load image
    img = mmcv.imread(img_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')

    # Create figure with 1x4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(osp.basename(img_path))

    data_samples = DetDataSample(
        metainfo={
            'img_id': 0,
            'img_path': img_path,
            'ori_shape': img.shape,
            'img_shape': img.shape,
            'scale_factor': np.array([1.0, 1.0]),
            'flip': False,
            'flip_direction': None
        }
    )

    model_maskrcnn.test_cfg.rpn.max_per_img = 100  # Limit number of proposals

    # Convert image to tensor and extract features
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to('cuda')  # Add batch dimension
    x = model_maskrcnn.extract_feat(img_tensor)

    # Get proposals using backbone features
    rpn_results_list = model_maskrcnn.rpn_head.predict(
        x,
        [data_samples],
        rescale=True
    )
    proposals = rpn_results_list[0]

    # Plot original image
    axes[0].imshow(img)
    for proposal in proposals:
        bboxes = proposal.bboxes.cpu().numpy()
        labels = proposal.labels.cpu().numpy()
        scores = proposal.scores.cpu().numpy()
        for bbox, label, score in zip(bboxes, labels, scores):
            x1, y1, x2, y2 = bbox
            if score > score_thr:
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                      fill=False, color='red', linewidth=1)
                axes[0].add_patch(rect)
    axes[0].set_title('Mask R-CNN Proposals')
    axes[0].axis('off')

    # Visualize Mask R-CNN detection results
    results = inference_detector(model_maskrcnn, img)
    detections = results.pred_instances[results.pred_instances.scores > score_thr]
    img_mask_rcnn_detections = draw_detections(img.copy(), detections, score_thr, model_maskrcnn.dataset_meta['classes'])
    axes[1].imshow(img_mask_rcnn_detections)
    axes[1].set_title('Mask R-CNN Detection Results')
    axes[1].axis('off')

    # Visualize Mask R-CNN instance segmentation results
    masks = detections.masks.cpu().numpy()
    labels = detections.labels.cpu().numpy()
    scores = detections.scores.cpu().numpy()
    img_mask_rcnn_segmentation = draw_segmentations_with_labels(img.copy(), masks, labels, scores, model_maskrcnn.dataset_meta['classes'])
    axes[2].imshow(img_mask_rcnn_segmentation)
    axes[2].set_title('Mask R-CNN Segmentation Results')
    axes[2].axis('off')

    # Visualize Sparse R-CNN detection results
    results = inference_detector(model_sparsercnn, img)
    detections = results.pred_instances[results.pred_instances.scores > score_thr]
    img_sparse_rcnn_detections = draw_detections(img.copy(), detections, score_thr, model_sparsercnn.dataset_meta['classes'])
    axes[3].imshow(img_sparse_rcnn_detections)
    axes[3].set_title('Sparse R-CNN Detection Results')
    axes[3].axis('off')

    # Save the figure
    plt.tight_layout()
    save_path = img_path.replace('.jpg', '_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Results saved to {save_path}')

def main():
    mask_rcnn_config = 'work_dirs/mask-rcnn_r50_fpn_1x_coco/mask-rcnn_r50_fpn_1x_coco.py'
    mask_rcnn_checkpoint = 'work_dirs/mask-rcnn_r50_fpn_1x_coco/epoch_12.pth'
    sparse_rcnn_config = 'work_dirs/sparse-rcnn_r50_fpn_1x_coco/sparse-rcnn_r50_fpn_1x_coco.py'
    sparse_rcnn_checkpoint = 'work_dirs/sparse-rcnn_r50_fpn_1x_coco/epoch_12.pth'

    model_maskrcnn = init_detector(mask_rcnn_config, mask_rcnn_checkpoint, device='cuda:0')
    model_sparsercnn = init_detector(sparse_rcnn_config, sparse_rcnn_checkpoint, device='cuda:0')

    # test_images = [
    #     'data/VOCdevkit/VOC2007/JPEGImages/000001.jpg',
    #     'data/VOCdevkit/VOC2007/JPEGImages/000002.jpg',
    #     'data/VOCdevkit/VOC2007/JPEGImages/000003.jpg',
    #     'data/VOCdevkit/VOC2007/JPEGImages/000004.jpg',
    # ]

    # test_images = [
    #     'data/VOCdevkit/coco/val2017/000017.jpg',
    #     'data/VOCdevkit/coco/val2017/000035.jpg',
    #     'data/VOCdevkit/coco/val2017/000026.jpg',
    #     'data/VOCdevkit/coco/val2017/000032.jpg',
    #     'data/VOCdevkit/coco/val2017/000036.jpg',
    #     'data/VOCdevkit/coco/val2017/000038.jpg',
    #     'data/VOCdevkit/coco/val2017/000039.jpg',
    # ]

    test_images = [
        'data/new_image/image_0017.jpg',
        'data/new_image/test_car.jpg',
        'data/new_image/image_0023.jpg',
        'data/new_image/image_0076.jpg',
        'data/new_image/image_0033.jpg',
    ]

    for img_path in test_images:
        visualize_proposals_and_results(model_maskrcnn, model_sparsercnn, img_path, 0.15)

if __name__ == '__main__':
    main()
