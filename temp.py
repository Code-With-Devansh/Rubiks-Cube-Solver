def color_to_face_mapping( raw):
        """
        Convert color-based string to URFDLB letters
        using center stickers as reference.
        """

        if len(raw) != 54:
            raise ValueError("Cube string must be 54 characters")

        # Extract centers (index 4 of each face block in FRBLDU order)
        centers = {
            raw[4]: 'F',
            raw[13]: 'R',
            raw[22]: 'B',
            raw[31]: 'L',
            raw[40]: 'D',
            raw[49]: 'U',
        }

        converted = ''.join(centers[c] for c in raw)
        print("color to face mapping: ", converted)
        return converted

color_to_face_mapping("OROWWYYYBYBBOYOYYGYWBWBBRYWOBRORGRRGWWGOOBGGBRRORGGWGW")