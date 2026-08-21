package com.avatarface.app.render

import android.graphics.Color

/**
 * Paleta del estilo «vector plano», portada de
 * `src/avatar_face/infrastructure/rendering/palette.py`. Los valores deben
 * coincidir exactamente con los de Python.
 */
object Palette {
    val backgrounds = mapOf(
        "coral" to "#F5907F",
        "mint" to "#79D6AF",
        "sky" to "#63C4EE",
        "lavender" to "#B4A3E4",
        "sand" to "#EEDCC0",
        "slate" to "#7C8B9C",
        "rose" to "#EFA8BE",
        "teal" to "#4FB3AE",
    )
    val skinTones = mapOf(
        "porcelain" to "#F6DCC9",
        "light" to "#F0C8A8",
        "beige" to "#E8B98F",
        "golden" to "#DCA771",
        "olive" to "#C69163",
        "tan" to "#BE8654",
        "brown" to "#96613D",
        "deep" to "#6E462F",
        "ebony" to "#4E3122",
    )
    val hairColors = mapOf(
        "black" to "#241F27",
        "brown" to "#5B3A26",
        "auburn" to "#8A3E27",
        "blonde" to "#DDB463",
        "blue" to "#31509E",
        "pink" to "#DC5FA5",
        "gray" to "#8D8D95",
        "red" to "#C0472A",
        "silver" to "#C8CBD2",
        "green" to "#3F8F5E",
    )
    val eyeColors = mapOf(
        "brown" to "#5C3A22",
        "blue" to "#3277AC",
        "green" to "#42804F",
        "gray" to "#5F6C76",
        "hazel" to "#8A6A32",
        "amber" to "#B47826",
    )
    val clothingColors = mapOf(
        "white" to "#F3F3F1",
        "charcoal" to "#3B4149",
        "red" to "#D6564C",
        "blue" to "#4A79BC",
        "green" to "#4FA177",
        "mustard" to "#DFAE45",
        "purple" to "#7C5FB0",
    )

    const val LINE = "#2A2229"
    const val SCLERA = "#FDFBF7"
    const val LIP = "#B9695F"
    const val TEETH = "#FDFBF7"
    const val GOLD = "#EEBC49"

    fun parse(hex: String): Int = Color.parseColor(hex)

    /** Interpola dos colores; `ratio` 0 devuelve el primero y 1 el segundo. */
    fun mix(first: Int, second: Int, ratio: Float): Int = Color.rgb(
        Math.round(Color.red(first) + (Color.red(second) - Color.red(first)) * ratio),
        Math.round(Color.green(first) + (Color.green(second) - Color.green(first)) * ratio),
        Math.round(Color.blue(first) + (Color.blue(second) - Color.blue(first)) * ratio),
    )

    /** Oscurece (`amount` < 0) o aclara (`amount` > 0) manteniendo el tono. */
    fun shade(color: Int, amount: Float): Int {
        val target = if (amount > 0) Color.rgb(255, 255, 255) else Color.rgb(0, 0, 0)
        return mix(color, target, kotlin.math.abs(amount))
    }
}
