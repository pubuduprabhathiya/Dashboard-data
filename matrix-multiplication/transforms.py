import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def matrix_transform(knl, context):
    knl = lp.split_iname(knl, "i", 32)
    knl = lp.split_iname(knl, "j", 32)

    knl = lp.tag_inames(
        knl,
        {
            "i_outer": "g.0",
            "i_inner": "l.0",
            "j_outer": "g.1",
            "j_inner": "l.1",
            "k": "for",
        },
    )

    return knl
