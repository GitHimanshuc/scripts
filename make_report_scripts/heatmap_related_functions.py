# %%
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Polygon


def get_domain_name(col_name, just_domain_names):
    if just_domain_names:
        return col_name

    def AMR_domains_to_decimal(subdoamin_name):
        # SphereC28.0.1
        a = subdoamin_name.split(".")
        # a = [SphereC28,0,1]
        decimal_rep = a[0] + "."
        # decimal_rep = SphereC28.
        for i in a[1:]:
            decimal_rep = decimal_rep + i
        # decimal_rep = SphereC28.01
        return decimal_rep

    if "on" in col_name:
        return AMR_domains_to_decimal(col_name.split(" ")[-1])
    if "in" in col_name:
        return AMR_domains_to_decimal(col_name.split(" ")[-1])
    elif "_" in col_name:
        return col_name.split("_")[0]
    elif "MinimumGridSpacing" in col_name:
        return col_name.split("[")[-1][:-1]
    else:
        raise Exception(
            f"{col_name} type not implemented in return_sorted_domain_names"
        )


def return_sorted_domain_names(
    domain_names, repeated_symmetric=False, num_Excision=2, just_domain_names=False
):
    # def filtered_domain_names(domain_names, filter):
    #   return [i for i in domain_names if get_domain_name(i).startswith(filter)]

    def filtered_domain_names(domain_names, filter):
        return [
            i
            for i in domain_names
            if re.match(filter, get_domain_name(i, just_domain_names))
        ]

    def sort_spheres(sphere_list, reverse=False):
        if len(sphere_list) == 0:
            return []
        if "SphereA" in sphere_list[0] and "Filled" not in sphere_list[0]:
            return sorted(
                sphere_list,
                key=lambda x: float(
                    get_domain_name(x, just_domain_names).lstrip("SphereA")
                ),
                reverse=reverse,
            )
        elif "SphereB" in sphere_list[0] and "Filled" not in sphere_list[0]:
            return sorted(
                sphere_list,
                key=lambda x: float(
                    get_domain_name(x, just_domain_names).lstrip("SphereB")
                ),
                reverse=reverse,
            )
        elif "SphereC" in sphere_list[0] and "Filled" not in sphere_list[0]:
            return sorted(
                sphere_list,
                key=lambda x: float(
                    get_domain_name(x, just_domain_names).lstrip("SphereC")
                ),
                reverse=reverse,
            )
        elif "SphereD" in sphere_list[0] and "Filled" not in sphere_list[0]:
            return sorted(
                sphere_list,
                key=lambda x: float(
                    get_domain_name(x, just_domain_names).lstrip("SphereD")
                ),
                reverse=reverse,
            )
        elif "SphereE" in sphere_list[0] and "Filled" not in sphere_list[0]:
            return sorted(
                sphere_list,
                key=lambda x: float(
                    get_domain_name(x, just_domain_names).lstrip("SphereE")
                ),
                reverse=reverse,
            )
        elif "FilledSphereA" in sphere_list[0]:
            return sorted(
                sphere_list,
                key=lambda x: float(
                    get_domain_name(x, just_domain_names).lstrip("FilledSphereA")
                ),
                reverse=reverse,
            )
        elif "FilledSphereB" in sphere_list[0]:
            return sorted(
                sphere_list,
                key=lambda x: float(
                    get_domain_name(x, just_domain_names).lstrip("FilledSphereB")
                ),
                reverse=reverse,
            )

    FilledCylinderCA = filtered_domain_names(domain_names, r"FilledCylinder.{0,2}CA")
    CylinderCA = filtered_domain_names(domain_names, r"Cylinder.{0,2}CA")
    FilledCylinderEA = filtered_domain_names(domain_names, r"FilledCylinder.{0,2}EA")
    CylinderEA = filtered_domain_names(domain_names, r"Cylinder.{0,2}EA")
    SphereA = sort_spheres(filtered_domain_names(domain_names, "SphereA"), reverse=True)
    FilledSphereA = sort_spheres(
        filtered_domain_names(domain_names, "FilledSphereA"), reverse=True
    )
    CylinderSMA = filtered_domain_names(domain_names, r"CylinderS.{0,2}MA")
    FilledCylinderMA = filtered_domain_names(domain_names, r"FilledCylinder.{0,2}MA")

    FilledCylinderMB = filtered_domain_names(domain_names, r"FilledCylinder.{0,2}MB")
    CylinderSMB = filtered_domain_names(domain_names, r"CylinderS.{0,2}MB")
    FilledSphereB = sort_spheres(
        filtered_domain_names(domain_names, "FilledSphereB"), reverse=True
    )
    SphereB = sort_spheres(filtered_domain_names(domain_names, "SphereB"), reverse=True)
    CylinderEB = filtered_domain_names(domain_names, r"Cylinder.{0,2}EB")
    FilledCylinderEB = filtered_domain_names(domain_names, r"FilledCylinder.{0,2}EB")
    CylinderCB = filtered_domain_names(domain_names, r"Cylinder.{0,2}CB")
    FilledCylinderCB = filtered_domain_names(domain_names, r"FilledCylinder.{0,2}CB")

    SphereC = sort_spheres(
        filtered_domain_names(domain_names, "SphereC"), reverse=False
    )
    SphereD = sort_spheres(
        filtered_domain_names(domain_names, "SphereD"), reverse=False
    )
    SphereE = sort_spheres(
        filtered_domain_names(domain_names, "SphereE"), reverse=False
    )

    import math

    if FilledSphereB == [] and repeated_symmetric:
        FilledSphereB = ["Excision"] * math.ceil(num_Excision / 2)
    if FilledSphereA == [] and repeated_symmetric:
        FilledSphereA = ["Excision"] * math.ceil(num_Excision / 2)

    combined_columns = [
        FilledCylinderCA,
        CylinderCA,
        FilledCylinderEA,
        CylinderEA,
        SphereA,
        FilledSphereA,
        CylinderSMA,
        FilledCylinderMA,
        FilledCylinderMB,
        CylinderSMB,
        SphereB,
        FilledSphereB,
        CylinderEB,
        FilledCylinderEB,
        CylinderCB,
        FilledCylinderCB,
        SphereC,
        SphereD,
        SphereE,
    ]
    if repeated_symmetric:
        combined_columns = [
            SphereE[::-1],
            SphereD[::-1],
            SphereC[::-1],
            FilledCylinderCA[::-1],
            CylinderCA[::-1],
            FilledCylinderEA[::-1],
            CylinderEA[::-1],
            SphereA,
            FilledSphereA,
            FilledSphereA[::-1],
            SphereA[::-1],
            CylinderSMA[::-1],
            FilledCylinderMA[::-1],
            FilledCylinderMB,
            CylinderSMB,
            SphereB,
            FilledSphereB,
            FilledSphereB[::-1],
            SphereB[::-1],
            CylinderEB,
            FilledCylinderEB,
            CylinderCB,
            FilledCylinderCB,
            SphereC,
            SphereD,
            SphereE,
        ]
    combined_columns = [item for sublist in combined_columns for item in sublist]

    # Just append the domains not following any patterns in the front. Mostly domains surrounding sphereA for high spin and mass ratios
    combined_columns_set = set(combined_columns)
    domain_names_set = set()
    for i in domain_names:
        domain_names_set.add(i)
    subdomains_not_sorted = list(domain_names_set - combined_columns_set)
    return subdomains_not_sorted + combined_columns


class BBH_domain_sym_ploy:
    def __init__(self, center_xA, rA, RA, rC, RC, nA, nC, color_dict: dict = None):
        self.center_xA = center_xA
        self.color_dict = color_dict
        self.rA = rA  # Largest SphereA radius
        self.RA = RA  # Radius of FilledCylinderE
        self.rC = rC  # Smallest SphereC radius
        self.RC = RC  # Radius of the largest SphereC

        self.nA = nA  # Number of SphereA
        self.nC = nC  # Number of SphereC

        self.alpha_for_FilledCylinderE_from_Center_bh = np.radians(50)
        self.outer_angle_for_CylinderSM_from_Center_bh = np.arccos(
            self.center_xA / self.RA
        )
        self.inner_angle_for_CylinderSM_from_Center_bh = (
            self.outer_angle_for_CylinderSM_from_Center_bh / 3
        )

        self.patches = []

        self.add_shpereCs()

        self.add_CylinderC(which_bh="A")
        self.add_FilledCylinderE(which_bh="A")
        self.add_CylinderE(which_bh="A")
        self.add_CylinderSM(which_bh="A")
        self.add_FilledCylinderM(which_bh="A")
        self.add_FilledCylinderC(which_bh="A")

        self.add_CylinderC(which_bh="B")
        self.add_FilledCylinderE(which_bh="B")
        self.add_CylinderE(which_bh="B")
        self.add_CylinderSM(which_bh="B")
        self.add_FilledCylinderM(which_bh="B")
        self.add_FilledCylinderC(which_bh="B")

        self.add_inner_shperes(which_bh="A")
        self.add_inner_shperes(which_bh="B")

        # print the unmatched domains
        print(self.color_dict)

    def get_matching_color(self, domain_name: str):
        if self.color_dict is None:
            return np.random.rand(
                3,
            )
        for key in self.color_dict.keys():
            if domain_name in key:
                # Remove the domain name from the key, this will allow us to see which domains were not matched
                return self.color_dict.pop(key)
        # No match found
        return "pink"

    def add_inner_shperes(self, which_bh):
        center = self.center_xA
        if which_bh == "B":
            center = -self.center_xA

        spheres_outer_radii = np.linspace(self.rA, 0, self.nA + 2)
        i = self.nA - 1
        for r in spheres_outer_radii[:-2]:
            domain_name = f"Sphere{which_bh}{i}"
            i = i - 1
            color = self.get_matching_color(domain_name)
            self.patches.append(
                Circle((center, 0), r, facecolor=color, edgecolor="black")
            )

        domain_name = f"Sphere{which_bh}{i}"
        i = i - 1
        color = self.get_matching_color(domain_name)
        self.patches.append(
            Circle(
                (center, 0),
                spheres_outer_radii[-2],
                facecolor="black",
                edgecolor="black",
            )
        )

    def add_shpereCs(self):
        spheres_outer_radii = np.linspace(self.RC, self.rC, self.nC + 1)[:-1]
        i = self.nC - 1
        for r in spheres_outer_radii:
            domain_name = f"SphereC{i}"
            i = i - 1
            color = self.get_matching_color(domain_name)
            self.patches.append(Circle((0, 0), r, facecolor=color, edgecolor="black"))

    def add_FilledCylinderE(self, which_bh):
        alpha = self.alpha_for_FilledCylinderE_from_Center_bh

        x_inner = self.center_xA + self.rA * np.cos(alpha)
        y_inner = self.rA * np.sin(alpha)
        x_outer = self.center_xA + self.RA * np.cos(alpha)
        y_outer = self.RA * np.sin(alpha)

        if which_bh == "B":
            x_inner = -x_inner
            x_outer = -x_outer
        vertices = [
            (x_inner, y_inner),
            (x_outer, y_outer),
            (x_outer, -y_outer),
            (x_inner, -y_inner),
        ]
        color = self.get_matching_color(f"FilledCylinderE{which_bh}")
        self.patches.append(
            Polygon(vertices, closed=True, facecolor=color, edgecolor="black")
        )

    def add_CylinderE(self, which_bh):
        alpha = self.alpha_for_FilledCylinderE_from_Center_bh
        beta = self.outer_angle_for_CylinderSM_from_Center_bh

        x_inner_away_from_center = self.center_xA + self.rA * np.cos(alpha)
        y_inner_away_from_center = self.rA * np.sin(alpha)
        x_outer_away_from_center = self.center_xA + self.RA * np.cos(alpha)
        y_outer_away_from_center = self.RA * np.sin(alpha)

        x_inner_closer_to_center = self.center_xA - self.rA * np.cos(beta)
        y_inner_closer_to_center = self.rA * np.sin(beta)
        x_outer_closer_to_center = 0
        y_outer_closer_to_center = self.RA * np.sin(beta)

        if which_bh == "B":
            x_inner_away_from_center = -x_inner_away_from_center
            x_outer_away_from_center = -x_outer_away_from_center
            x_inner_closer_to_center = -x_inner_closer_to_center
            x_outer_closer_to_center = -x_outer_closer_to_center

        vertices = [
            (x_inner_away_from_center, y_inner_away_from_center),
            (x_outer_away_from_center, y_outer_away_from_center),
            (x_outer_closer_to_center, y_outer_closer_to_center),
            (x_inner_closer_to_center, y_inner_closer_to_center),
            (x_inner_closer_to_center, -y_inner_closer_to_center),
            (x_outer_closer_to_center, -y_outer_closer_to_center),
            (x_outer_away_from_center, -y_outer_away_from_center),
            (x_inner_away_from_center, -y_inner_away_from_center),
        ]
        color = self.get_matching_color(f"CylinderE{which_bh}")
        self.patches.append(
            Polygon(vertices, closed=True, facecolor=color, edgecolor="black")
        )

    def add_CylinderC(self, which_bh):
        alpha = self.alpha_for_FilledCylinderE_from_Center_bh
        beta = self.outer_angle_for_CylinderSM_from_Center_bh

        x_inner_away_from_center = self.center_xA + self.rA * np.cos(alpha)
        y_inner_away_from_center = self.rA * np.sin(alpha)
        x_outer_away_from_center = self.rC * np.cos(np.radians(30))
        y_outer_away_from_center = self.rC * np.sin(np.radians(30))

        x_inner_closer_to_center = 0
        y_inner_closer_to_center = self.RA * np.sin(beta)
        x_outer_closer_to_center = 0
        y_outer_closer_to_center = self.rC

        if which_bh == "B":
            x_inner_away_from_center = -x_inner_away_from_center
            x_outer_away_from_center = -x_outer_away_from_center
            x_inner_closer_to_center = -x_inner_closer_to_center
            x_outer_closer_to_center = -x_outer_closer_to_center

        vertices = [
            (x_inner_closer_to_center, y_inner_closer_to_center),
            (x_outer_closer_to_center, y_outer_closer_to_center),
            (x_outer_away_from_center, y_outer_away_from_center),
            (x_inner_away_from_center, y_inner_away_from_center),
            (x_inner_away_from_center, -y_inner_away_from_center),
            (x_outer_away_from_center, -y_outer_away_from_center),
            (x_outer_closer_to_center, -y_outer_closer_to_center),
            (x_inner_closer_to_center, -y_inner_closer_to_center),
        ]
        color = self.get_matching_color(f"CylinderC{which_bh}")
        self.patches.append(
            Polygon(vertices, closed=True, facecolor=color, edgecolor="black")
        )

    def add_CylinderSM(self, which_bh):
        beta = self.outer_angle_for_CylinderSM_from_Center_bh
        gamma = self.inner_angle_for_CylinderSM_from_Center_bh

        x_inner_away_from_center = self.center_xA - self.rA * np.cos(beta)
        y_inner_away_from_center = self.rA * np.sin(beta)
        x_outer_away_from_center = 0
        y_outer_away_from_center = self.RA * np.sin(beta)

        x_inner_closer_to_center = self.center_xA - self.rA * np.cos(gamma)
        y_inner_closer_to_center = self.rA * np.sin(gamma)
        x_outer_closer_to_center = 0
        y_outer_closer_to_center = self.RA * np.sin(gamma)

        if which_bh == "B":
            x_inner_away_from_center = -x_inner_away_from_center
            x_outer_away_from_center = -x_outer_away_from_center
            x_inner_closer_to_center = -x_inner_closer_to_center
            x_outer_closer_to_center = -x_outer_closer_to_center

        vertices = [
            (x_inner_away_from_center, y_inner_away_from_center),
            (x_outer_away_from_center, y_outer_away_from_center),
            (x_outer_closer_to_center, y_outer_closer_to_center),
            (x_inner_closer_to_center, y_inner_closer_to_center),
            (x_inner_closer_to_center, -y_inner_closer_to_center),
            (x_outer_closer_to_center, -y_outer_closer_to_center),
            (x_outer_away_from_center, -y_outer_away_from_center),
            (x_inner_away_from_center, -y_inner_away_from_center),
        ]
        color = self.get_matching_color(f"CylinderSM{which_bh}")
        self.patches.append(
            Polygon(vertices, closed=True, facecolor=color, edgecolor="black")
        )

    def add_FilledCylinderM(self, which_bh):
        gamma = self.inner_angle_for_CylinderSM_from_Center_bh

        x_inner = self.center_xA - self.rA * np.cos(gamma)
        y_inner = self.rA * np.sin(gamma)
        x_outer = 0
        y_outer = self.RA * np.sin(gamma)

        if which_bh == "B":
            x_inner = -x_inner
            x_outer = -x_outer
        vertices = [
            (x_inner, y_inner),
            (x_outer, y_outer),
            (x_outer, -y_outer),
            (x_inner, -y_inner),
        ]
        color = self.get_matching_color(f"FilledCylinderM{which_bh}")
        self.patches.append(
            Polygon(vertices, closed=True, facecolor=color, edgecolor="black")
        )

    def add_FilledCylinderC(self, which_bh):
        alpha = self.alpha_for_FilledCylinderE_from_Center_bh

        x_inner = self.center_xA + self.RA * np.cos(alpha)
        y_inner = self.RA * np.sin(alpha)
        x_outer = self.rC * np.cos(np.radians(30))
        y_outer = self.rC * np.sin(np.radians(30))

        if which_bh == "B":
            x_inner = -x_inner
            x_outer = -x_outer
        vertices = [
            (x_inner, y_inner),
            (x_outer, y_outer),
            (x_outer, -y_outer),
            (x_inner, -y_inner),
        ]
        color = self.get_matching_color(f"FilledCylinderC{which_bh}")
        self.patches.append(
            Polygon(vertices, closed=True, facecolor=color, edgecolor="black")
        )


def scalar_to_color(scalar_dict, min_max_tuple=None, color_map="viridis"):
    arr_keys, arr_vals = [], []
    for key, val in scalar_dict.items():
        if np.isnan(val):
            continue
        else:
            arr_keys.append(key)
            arr_vals.append(val)

    scalar_array = np.array(arr_vals, dtype=np.float64)
    scalar_array = np.log10(scalar_array)
    min_val = np.min(scalar_array)
    max_val = np.max(scalar_array)
    print(min_val, max_val)
    if min_max_tuple is not None:
        min_val, max_val = min_max_tuple
    scalar_normalized = (scalar_array - min_val) / (max_val - min_val)

    colormap = plt.get_cmap(color_map)
    colors = {}
    for key, value in zip(arr_keys, scalar_normalized):
        colors[key] = colormap(value)

    # Get colorbar
    norm = Normalize(vmin=min_val, vmax=max_val)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    return colors, sm
