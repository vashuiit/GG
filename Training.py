for epoch in range(num_epochs):
    for content_img, style_img in train_dataloader:
        content_outputs, style_outputs = model(content_img)
        content_loss = calculate_content_loss(content_outputs, content_img)
        style_loss = calculate_style_loss(style_outputs, style_img)
        total_loss = content_loss + style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()