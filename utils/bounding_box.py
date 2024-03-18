import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    Usage:
        This class represents a set of bounding boxes.
        The bounding boxes are represented as a Nx4 Tensor.
        In order to uniquely determine the bounding boxes with respect
        to an image, we also store the corresponding image dimensions.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode

    def convert(self, mode):
        """
        Args:
            mode : xyxy xywh
        """
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            bbox = torch.cat(
                ((xmin + xmax) / 2, (ymin + ymax) / 2, (xmax - xmin), (ymax - ymin)), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            xc, yc, w, h = self.bbox.split(1, dim=-1)
            return (
                xc - 0.5 * w,
                yc - 0.5 * h,
                xc + 0.5 * w,
                yc + 0.5 * h
            )
        else:
            raise RuntimeError("Should not be here")
    
    def shift(self, padded_size, left : int, top : int):
        """
        Returns a shifted copy of this bounding box
        params:
            left : xshift, top : yshift
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        shifted_xmin, shifted_xmax = xmin + left, xmax + left
        shifted_ymin, shifted_ymax = ymin + top, ymax + top
        shifted_box = torch.cat(
            (shifted_xmin, shifted_ymin, shifted_xmax, shifted_ymax), dim=-1
        )
        bbox = BoxList(shifted_box, padded_size, mode="xyxy")
        return bbox.convert(self.mode)


    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxy")

        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            transposed_xmin = image_width - xmax
            transposed_xmax = image_width - xmin
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        return bbox.convert(self.mode)

    def check_crop_valid(self, region):
        """
        box : [x_min, y_min, w, h]
        """
        rymin, rxmin, h, w = region
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        cropped_xmin = (xmin - rxmin).clamp(min=0, max=w)
        cropped_ymin = (ymin - rymin).clamp(min=0, max=h)
        cropped_xmax = (xmax - rxmin).clamp(min=0, max=w)
        cropped_ymax = (ymax - rymin).clamp(min=0, max=h)
        
        valid = not any((cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)) 

        return valid

    def crop(self, region):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        rymin, rxmin, h, w = region
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        cropped_xmin = (xmin - rxmin).clamp(min=0, max=w)
        cropped_ymin = (ymin - rymin).clamp(min=0, max=h)
        cropped_xmax = (xmax - rxmin).clamp(min=0, max=w)
        cropped_ymax = (ymax - rymin).clamp(min=0, max=h)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        return bbox.convert(self.mode)

    def normalize(self):
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        image_width, image_height = self.size
        xmin = xmin / image_width
        ymin = ymin / image_height
        xmax = xmax / image_width
        ymax = ymax / image_height
        normalized_bbox = torch.cat(
            (xmin, ymin, xmax, ymax), dim=-1
        )
        bbox = BoxList(normalized_bbox, self.size, mode="xyxy")
        return bbox.convert("xywh")

    # Tensor-like methods
    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy(self):
        return BoxList(self.bbox, self.size, self.mode)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)
