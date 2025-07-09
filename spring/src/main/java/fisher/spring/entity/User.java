package fisher.spring.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import lombok.Data;
import lombok.ToString;

@Data
@ToString
public class User {
    @TableId(type = IdType.AUTO)
    private Long userId;
    private String name;
    private Integer age;
    private String email;
    @TableLogic
    private int isDelete;

}

