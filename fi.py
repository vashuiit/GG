def adapt_style(model, content_img, style_reference_img):
    content_outputs, style_outputs = model(content_img)
    adapted_img = apply_style(content_img, style_outputs_reference=style_outputs)

    return adapted_img
