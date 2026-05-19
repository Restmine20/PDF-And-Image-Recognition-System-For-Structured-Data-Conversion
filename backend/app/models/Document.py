from app.settings import Base, Column, String, ForeignKey
from sqlalchemy import UUID
import uuid


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_filename = Column(String(255), nullable=False)
    batch_id = Column(UUID(as_uuid=True), ForeignKey('batches.id'), nullable=False)

    filename = Column(String(500), nullable=False)
    result_filename = Column(String(500), nullable=False)

    task_id = Column(String(255), nullable=False)
    status = Column(String(50), default='PENDING')
