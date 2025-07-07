package fisher.mybatis.pojo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.io.Serializable;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Student implements Serializable {
    private long studentId;
    private String studentName;
    private String studentBirth;
    private String studentAddress;
    private long studentNo;
    private long gradeId;
    private int studentAge;
}
