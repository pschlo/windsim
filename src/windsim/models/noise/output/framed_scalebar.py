import warnings

from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox, HPacker,
                                  TextArea, VPacker)
from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar


class FramedScaleBar(ScaleBar):
   def draw(self, renderer, *args, **kwargs):
        if not self.get_visible():
            return
        if self.dx == 0:
            return

        # Late import
        from matplotlib import rcParams

        # Deprecation
        if rcParams.get("scalebar.height_fraction") is not None:
            warnings.warn(
                "The scalebar.height_fraction parameter in matplotlibrc is deprecated. "
                "Use scalebar.width_fraction instead.",
                DeprecationWarning,
            )
            rcParams.setdefault(
                "scalebar.width_fraction", rcParams["scalebar.height_fraction"]
            )

        # Get parameters
        def _get_value(attr, default):
            value = getattr(self, attr)
            if value is None:
                value = rcParams.get("scalebar." + attr, default)
            return value

        length_fraction = _get_value("length_fraction", 0.2)
        width_fraction = _get_value("width_fraction", 0.01)
        location = _get_value("location", "upper right")
        if isinstance(location, str):
            location = self._LOCATIONS[location.lower()]
        pad = _get_value("pad", 0.2)
        border_pad = _get_value("border_pad", 0.1)
        sep = _get_value("sep", 5)
        frameon = _get_value("frameon", True)
        color = _get_value("color", "k")
        box_color = _get_value("box_color", "w")
        box_alpha = _get_value("box_alpha", 1.0)
        scale_loc = _get_value("scale_loc", "bottom").lower()
        label_loc = _get_value("label_loc", "top").lower()
        font_properties = self.font_properties
        fixed_value = self.fixed_value
        fixed_units = self.fixed_units or self.units
        rotation = _get_value("rotation", "horizontal").lower()
        label = self.label

        # Create text properties
        textprops = {"color": color, "rotation": rotation}
        if font_properties is not None:
            textprops["fontproperties"] = font_properties

        # Calculate value, units and length
        ax = self.axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if rotation == "vertical":
            xlim, ylim = ylim, xlim

        # Mode 1: Auto
        if self.fixed_value is None:
            length_px = abs(xlim[1] - xlim[0]) * length_fraction
            length_px, value, units = self._calculate_best_length(length_px)

        # Mode 2: Fixed
        else:
            value = fixed_value
            units = fixed_units
            length_px = self._calculate_exact_length(value, units)

        scale_text = self.scale_formatter(value, self.dimension.to_latex(units))

        width_px = abs(ylim[1] - ylim[0]) * width_fraction

        # Create scale bar
        if rotation == "horizontal":
            scale_rect = Rectangle(
                (0, 0),
                length_px,
                width_px,
                fill=True,
                facecolor=color,
                edgecolor="none",
            )
        else:
            scale_rect = Rectangle(
                (0, 0),
                width_px,
                length_px,
                fill=True,
                facecolor=color,
                edgecolor="none",
            )

        scale_bar_box = AuxTransformBox(ax.transData)
        scale_bar_box.add_artist(scale_rect)

        # Create scale text
        if scale_loc != "none":
            scale_text_box = TextArea(scale_text, textprops=textprops)

            if scale_loc in ["bottom", "right"]:
                children = [scale_bar_box, scale_text_box]
            else:
                children = [scale_text_box, scale_bar_box]

            if scale_loc in ["bottom", "top"]:
                Packer = VPacker
            else:
                Packer = HPacker

            scale_box = Packer(children=children, align="center", pad=0, sep=sep)

        else:
            scale_box = scale_bar_box

        # Create label
        if label and label_loc != "none":
            label_box = TextArea(label, textprops=textprops)
        else:
            label_box = None

        # Create final offset box
        if label_box:
            if label_loc in ["bottom", "right"]:
                children = [scale_box, label_box]
            else:
                children = [label_box, scale_box]

            if label_loc in ["bottom", "top"]:
                Packer = VPacker
            else:
                Packer = HPacker

            child = Packer(children=children, align="center", pad=0, sep=sep)
        else:
            child = scale_box

        box = AnchoredOffsetbox(
            loc=location, pad=pad, borderpad=border_pad, child=child, frameon=frameon
        )

        box.axes = ax
        box.set_figure(self.get_figure())
        box.patch.set_color(box_color)
        box.patch.set_alpha(box_alpha)
        box.patch.set_edgecolor('black')
        box.patch.set_linewidth(1)
        box.draw(renderer)
