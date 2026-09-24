# Technical Design Document for Handling Multiple Shipping Addresses in Orders

**Issue:** Payment authorization is failing for orders that have multiple shipping addresses.

## Affected Services

### orders

Complexity: medium | Data model: CustomerOrder

The existing logic in OrdersController does not accommodate multiple shipping addresses, which are essential for the flexibility of order processing in customer purchases. By activating multiple address handling, the system becomes more robust for diverse customer needs.

- `src/main/java/works/weave/socks/orders/controllers/OrdersController.java` — **OrdersController.newOrder** _(grounded in real source)_
  - Change: Modify `newOrder` method to handle multiple addresses for payment requests.
  - Implementation: Modify the payment handling section within `newOrder` to iterate over a list of addresses. Create separate PaymentRequest objects for each address and ensure successful payment authorization for at least one of them before proceeding with order creation.
  - Proposed approach (may reflect real source shown to the model):
    ```
    for each address in item.addresses:
        paymentRequest = new PaymentRequest(address, card, customer, amount)
        if paymentService.authorise(paymentRequest):
            successfulPayment = true
            break
    if not successfulPayment:
        throw new PaymentDeclinedException('Payment authorization failed for all provided addresses.')
    ```
  - Why: Adjusting the 'newOrder' method to process multiple shipping addresses for payment is necessary due to the current logical handling which assumes a single address. This change ensures at least one address in multi-address orders is successfully authorized, thus improving order processing reliability.
  - Acceptance criteria:
    - Run tests to confirm that orders with one or more addresses successfully authorize payment.
    - Ensure that the order is created only if at least one address passes payment authorization.
    - Verify that payment failure for all addresses results in a PaymentDeclinedException.

## Risks

- If the logic for handling multiple addresses introduces errors, it could lead to incorrect order processing or failure to authorize payments.

## Testing Notes

Use unit tests to simulate multi-address orders and verify successful payment authorization and order creation. Check both scenarios where at least one address succeeds and all fail.

## Open Questions

- How should multi-address orders be reflected in order confirmations and customer notifications?
- Are there any dependencies on the payment service that might require modifications to authorize multiple addresses?

## Overall Reasoning

To resolve payment authorization failures in orders with multiple addresses, it is critical to adjust the Orders service to support multiple addresses. This involves iterating through addresses for payment authorization attempts and ensuring at least one is successful before the order is confirmed. This method enhances reliability and customer satisfaction by efficiently managing multiple shipment scenarios.