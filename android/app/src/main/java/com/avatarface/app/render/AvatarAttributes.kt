package com.avatarface.app.render

import org.json.JSONObject

/**
 * Atributos del avatar, portados de `src/avatar_face/domain/attributes.py`.
 *
 * Los nueve primeros son los del vocabulario heredado y el resto se añadió con
 * el ADR 0012; los valores por defecto coinciden con `DEFAULT_ATTRIBUTES`.
 */
data class AvatarAttributes(
    val expression: String = "calm",
    val faceShape: String = "oval",
    val skinTone: String = "light",
    val hairStyle: String = "short",
    val hairColor: String = "brown",
    val eyeColor: String = "brown",
    val eyeShape: String = "almond",
    val accessory: String = "none",
    val background: String = "sky",
    val browStyle: String = "natural",
    val noseStyle: String = "straight",
    val facialHair: String = "none",
    val glasses: String = "none",
    val earrings: String = "none",
    val freckles: String = "none",
    val clothing: String = "crew neck",
    val clothingColor: String = "blue",
) {
    /** Gafas pedidas de forma explícita o a través del atributo heredado. */
    val effectiveGlasses: String
        get() = when {
            glasses != "none" -> glasses
            accessory == "round glasses" -> "round"
            accessory == "square glasses" -> "square"
            accessory == "sunglasses" -> "sunglasses"
            else -> "none"
        }

    val effectiveEarrings: String
        get() = when {
            earrings != "none" -> earrings
            accessory == "earrings" -> "studs"
            else -> "none"
        }

    val effectiveFreckles: String
        get() = when {
            freckles != "none" -> freckles
            accessory == "freckles" -> "light"
            else -> "none"
        }

    companion object {
        /** Lee los atributos de un objeto JSON, usando los valores por defecto. */
        fun fromJson(json: JSONObject): AvatarAttributes {
            val base = AvatarAttributes()
            fun value(key: String, fallback: String) = json.optString(key, fallback)
            return AvatarAttributes(
                expression = value("expression", base.expression),
                faceShape = value("face_shape", base.faceShape),
                skinTone = value("skin_tone", base.skinTone),
                hairStyle = value("hair_style", base.hairStyle),
                hairColor = value("hair_color", base.hairColor),
                eyeColor = value("eye_color", base.eyeColor),
                eyeShape = value("eye_shape", base.eyeShape),
                accessory = value("accessory", base.accessory),
                background = value("background", base.background),
                browStyle = value("brow_style", base.browStyle),
                noseStyle = value("nose_style", base.noseStyle),
                facialHair = value("facial_hair", base.facialHair),
                glasses = value("glasses", base.glasses),
                earrings = value("earrings", base.earrings),
                freckles = value("freckles", base.freckles),
                clothing = value("clothing", base.clothing),
                clothingColor = value("clothing_color", base.clothingColor),
            )
        }
    }
}
